#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define PROVIDES_EXECUTORS
#include <boost/exception_ptr.hpp>
#include <boost/optional.hpp>

#include <boost/thread/executors/basic_thread_pool.hpp>

#include <folly/Unit.h>
#include <folly/futures/Future.h>

#include <clipper/containers.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/exceptions.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/query_processor.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/timers.hpp>

#define UNREACHABLE() assert(false)

using std::tuple;
using std::vector;
using zmq::context_t;
using zmq::socket_t;
using zmq::message_t;

namespace clipper {

QueryProcessor::QueryProcessor() : state_db_(std::make_shared<StateDB>()), context(1), send_sock(context, ZMQ_PAIR),
                                   rcv_sock(context, ZMQ_PAIR) {
  send_sock.bind("tcp://*:8080");
  rcv_sock.bind("tcp://*:8083");
  selection_policies_.emplace(DefaultOutputSelectionPolicy::get_name(),
                              std::make_shared<DefaultOutputSelectionPolicy>());
  log_info(LOGGING_TAG_QUERY_PROCESSOR, "Query Processor started");
}

std::shared_ptr<StateDB> QueryProcessor::get_state_table() const {
  return state_db_;
}

folly::Future<Response> QueryProcessor::predict(Query query) {
  clipper::Config& conf = clipper::get_config();
  long query_id = query_counter_.fetch_add(1);
  std::string query_json = (query.get_json_string("select") + std::to_string(query_id) + "}");
  message_t msg(query_json.length());
  memcpy ( (void *) msg.data(), query_json.c_str(), query_json.length());
  std::string cmd = "python clipper/selection_policy_testing/selection_frontend.py "
                    + conf.get_redis_address() + " " + std::to_string(conf.get_redis_port()) + " &";
  popen(cmd.c_str(), "r");
  send_sock.send(msg);
  message_t responsem;
  rcv_sock.recv(&responsem);
  long q_id = std::stol(std::string(static_cast<char*>(responsem.data()), responsem.size()));
  int more;
  size_t more_size = sizeof(more);
  rcv_sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
  std::vector<VersionedModelId> candidate_model_ids;
  while (more) {
    rcv_sock.recv(&responsem);
    std::string name = std::string(static_cast<char*>(responsem.data()), responsem.size());
    rcv_sock.recv(&responsem);
    std::string id = std::string(static_cast<char*>(responsem.data()), responsem.size());
    candidate_model_ids.push_back(VersionedModelId(name, id));
    rcv_sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
  }
  std::vector<PredictTask> tasks;
  for(std::vector<VersionedModelId>::iterator it = candidate_model_ids.begin(); it != candidate_model_ids.end(); ++it) {
    tasks.emplace_back(query.input_, *it, 1.0, q_id, query.latency_budget_micros_);
  }

  boost::optional<std::string> default_explanation;

  log_info_formatted(LOGGING_TAG_QUERY_PROCESSOR, "Found {} tasks",
                     tasks.size());

  vector<folly::Future<Output>> task_futures =
      task_executor_.schedule_predictions(tasks);
  if (task_futures.empty()) {
    default_explanation = "No connected models found for query";
    log_error_formatted(LOGGING_TAG_QUERY_PROCESSOR,
                        "No connected models found for query with id: {}",
                        query_id);
  }

  size_t num_tasks = task_futures.size();
  folly::Future<folly::Unit> timer_future =
      timer_system_.set_timer(query.latency_budget_micros_);

  std::shared_ptr<std::mutex> outputs_mutex = std::make_shared<std::mutex>();
  std::vector<Output> outputs;
  outputs.reserve(task_futures.size());
  std::shared_ptr<std::vector<Output>> outputs_ptr =
      std::make_shared<std::vector<Output>>(std::move(outputs));
  std::vector<folly::Future<folly::Unit>> wrapped_task_futures;
  for (auto it = task_futures.begin(); it < task_futures.end(); it++) {
    wrapped_task_futures.push_back(
        it->then([outputs_mutex, outputs_ptr](Output output) {
            std::lock_guard<std::mutex> lock(*outputs_mutex);
            outputs_ptr->push_back(output);
          }).onError([](const std::exception& e) {
          log_error_formatted(
              LOGGING_TAG_QUERY_PROCESSOR,
              "Unexpected error while executing prediction tasks: {}",
              e.what());
        }));
  }

  folly::Future<folly::Unit> all_tasks_completed_future =
      folly::collect(wrapped_task_futures)
          .then([](std::vector<folly::Unit> /* outputs */) {});

  std::vector<folly::Future<folly::Unit>> when_either_futures;
  when_either_futures.push_back(std::move(all_tasks_completed_future));
  when_either_futures.push_back(std::move(timer_future));

  folly::Future<std::pair<size_t, folly::Try<folly::Unit>>>
      response_ready_future = folly::collectAny(when_either_futures);

  folly::Promise<Response> response_promise;
  folly::Future<Response> response_future = response_promise.getFuture();

  response_ready_future.then([
    this, candidate_model_ids, outputs_ptr, outputs_mutex, num_tasks, query, query_id, response_promise = std::move(response_promise),
    default_explanation
  ](const std::pair<size_t,
                    folly::Try<folly::Unit>>& /* completed_future */) mutable {
      std::lock_guard<std::mutex> outputs_lock(*outputs_mutex);
      if (outputs_ptr->empty() && num_tasks > 0 && !default_explanation) {
      default_explanation =
          "Failed to retrieve a prediction response within the specified "
          "latency SLO";
    }
    std::string response_json = "{\"query_id\":" + std::to_string(query_id) + ", \"msg\": \"combine\","
                              + " \"selection_policy\": \"" + query.selection_policy_ + "\", \"model_outputs\":[";
    int i = 1;
    for (auto outputi : *outputs_ptr) {
      response_json += outputi.get_y_hat_string();
      if (i != num_tasks) {
        response_json += ", ";
      }
      i++;
    }
    response_json += "]}";
    message_t msg(response_json.length());
    memcpy ( (void *) msg.data(), response_json.c_str(), response_json.length());
    send_sock.send(msg);
    message_t responsem;
    rcv_sock.recv(&responsem);
    CombinedOutput final_output{std::string(static_cast<char*>(responsem.data()), responsem.size())};
    std::chrono::time_point<std::chrono::high_resolution_clock> end =
        std::chrono::high_resolution_clock::now();
    long duration_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end - query.create_time_)
            .count();
    std::string models = "[";
    for (std::vector<VersionedModelId>::iterator i = candidate_model_ids.begin(); i != candidate_model_ids.end(); i++) {
      models += i->get_json_string();
      if (i + 1 != candidate_model_ids.end()) {
          models += ", ";
        }
    }
    models += "]";
      Response response{query,
                      query_id,
                      final_output,
                      duration_micros,
                      models,
                      default_explanation};
    response_promise.setValue(response);
  });
  return response_future;
}

folly::Future<FeedbackAck> QueryProcessor::update(FeedbackQuery feedback) {
  log_info(LOGGING_TAG_QUERY_PROCESSOR, "Received feedback for user {}",
           feedback.user_id_);

  long query_id = query_counter_.fetch_add(1);
  folly::Future<FeedbackAck> error_response = folly::makeFuture(false);

  std::string query_json = feedback.get_json_string("feedback-select");
  message_t msg(query_json.length());
  memcpy ( (void *) msg.data(), query_json.c_str(), query_json.length());
  send_sock.send(msg);
  message_t responsem;
  rcv_sock.recv(&responsem);
  long q_id = std::stol(std::string(static_cast<char*>(responsem.data()), responsem.size()));
  int more;
  size_t more_size = sizeof(more);
  rcv_sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
  std::vector<VersionedModelId> candidate_model_ids;
  while (more) {
    rcv_sock.recv(&responsem);
    std::string name = std::string(static_cast<char*>(responsem.data()), responsem.size());
    rcv_sock.recv(&responsem);
    std::string id = std::string(static_cast<char*>(responsem.data()), responsem.size());
    candidate_model_ids.push_back(VersionedModelId(name, id));
    rcv_sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
  }

  std::vector<PredictTask> tasks;
  for(std::vector<VersionedModelId>::iterator it = candidate_model_ids.begin(); it != candidate_model_ids.end(); ++it) {
    // latency micros?
    tasks.emplace_back(feedback.feedback_.input_, *it, 1.0, q_id, 1000000);
  }

  log_info_formatted(LOGGING_TAG_QUERY_PROCESSOR,
                     "Scheduling {} prediction tasks",
                     tasks.size());

  vector<folly::Future<Output>> task_futures =
      task_executor_.schedule_predictions(tasks);
  if (task_futures.empty()) {
    // ...
  }

  size_t num_tasks = task_futures.size();
  std::shared_ptr<std::mutex> outputs_mutex = std::make_shared<std::mutex>();
  std::vector<Output> outputs;
  outputs.reserve(task_futures.size());
  std::shared_ptr<std::vector<Output>> outputs_ptr =
      std::make_shared<std::vector<Output>>(std::move(outputs));
  std::vector<folly::Future<folly::Unit>> wrapped_task_futures;
  for (auto it = task_futures.begin();
            it < task_futures.end(); it++) {
    wrapped_task_futures.push_back(
        it->then([outputs_mutex, outputs_ptr](Output output) {
            std::lock_guard<std::mutex> lock(*outputs_mutex);
            outputs_ptr->push_back(output);
          }).onError([](const std::exception& e) {
          log_error_formatted(
              LOGGING_TAG_QUERY_PROCESSOR,
              "Unexpected error while executing predicton tasks: {}",
              e.what());
        }));
  }

  folly::Future<std::vector<folly::Unit>> all_tasks_completed =
      folly::collect(wrapped_task_futures);

  folly::Promise<FeedbackAck> select_policy_update_promise;
  folly::Future<FeedbackAck> select_policy_updated =
      select_policy_update_promise.getFuture();
  auto state_table = get_state_table();

  all_tasks_completed.then([
    this, query_id, outputs_ptr, outputs_mutex, num_tasks, state_table, feedback,
    select_policy_update_promise = std::move(select_policy_update_promise)
  ](const std::vector<folly::Unit>& /* completed_future */) mutable {
    std::lock_guard<std::mutex> outputs_lock(*outputs_mutex);
    std::string response_json = "{\"query_id\":" + std::to_string(query_id) +
            ", \"msg\": \"update\", \"selection_policy\": \"" + feedback.selection_policy_ +
            "\", \"model_outputs\":[";
    int i = 1;
    for (auto outputi : *outputs_ptr) {
      response_json += outputi.get_y_hat_string();
      if (i != num_tasks) {
        response_json += ", ";
      }
      ++i;
    }
    response_json += "]}";
    message_t msg(response_json.length());
    memcpy(msg.data(), response_json.c_str(), response_json.length());
    send_sock.send(msg);

    message_t responsem;
    rcv_sock.recv(&responsem);

    // save responsem.data()
    select_policy_update_promise.setValue(true);
  });

  return select_policy_updated;
}

}  // namespace clipper
