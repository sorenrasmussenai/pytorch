#include <c10d/test/StoreTestCommon.hpp>

#include <cstdlib>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include <c10d/PrefixStore.hpp>
#include <c10d/TCPStore.hpp>

std::condition_variable cv;
std::mutex cvMutex;
constexpr int64_t kShortStoreTimeoutMillis = 100;
constexpr int64_t kStoreCallbackTimeoutMillis = 3000;
constexpr int defaultTimeout = 20;

c10::intrusive_ptr<c10d::TCPStore> _createServer(
    int numWorkers = 1,
    int timeout = defaultTimeout) {
  return c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      0,
      numWorkers,
      true,
      std::chrono::seconds(timeout),
      /* wait */ false);
}

// Different ports for different tests.
void testHelper(const std::string& prefix = "") {
  const auto numThreads = 16;
  const auto numWorkers = numThreads + 1;

  auto serverTCPStore = _createServer(numWorkers);

  auto serverStore =
      c10::make_intrusive<c10d::PrefixStore>(prefix, serverTCPStore);
  // server store
  auto serverThread = std::thread([&serverStore, &serverTCPStore] {
    // Wait for all workers to join.
    serverTCPStore->waitForWorkers();

    // Basic set/get on the server store
    c10d::test::set(*serverStore, "key0", "value0");
    c10d::test::set(*serverStore, "key1", "value1");
    c10d::test::set(*serverStore, "key2", "value2");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::check(*serverStore, "key1", "value1");
    c10d::test::check(*serverStore, "key2", "value2");
    serverStore->add("counter", 1);
    auto numKeys = serverStore->getNumKeys();
    // We expect 5 keys since 3 are added above, 'counter' is added by the
    // helper thread, and the init key to coordinate workers.
    EXPECT_EQ(numKeys, 5);

    // Check compareSet, does not check return value
    c10d::test::compareSet(
        *serverStore, "key0", "wrongCurrentValue", "newValue");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::compareSet(*serverStore, "key0", "value0", "newValue");
    c10d::test::check(*serverStore, "key0", "newValue");

    auto delSuccess = serverStore->deleteKey("key0");
    // Ensure that the key was successfully deleted
    EXPECT_TRUE(delSuccess);
    auto delFailure = serverStore->deleteKey("badKeyName");
    // The key was not in the store so the delete operation should have failed
    // and returned false.
    EXPECT_FALSE(delFailure);
    numKeys = serverStore->getNumKeys();
    EXPECT_EQ(numKeys, 4);
    auto timeout = std::chrono::milliseconds(kShortStoreTimeoutMillis);
    serverStore->setTimeout(timeout);
    EXPECT_THROW(serverStore->get("key0"), std::runtime_error);
  });

  // Hammer on TCPStore
  std::vector<std::thread> threads;
  const auto numIterations = 1000;
  c10d::test::Semaphore sem1, sem2;

  // Each thread will have a client store to send/recv data
  std::vector<c10::intrusive_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<c10::intrusive_ptr<c10d::PrefixStore>> clientStores;
  for (auto i = 0; i < numThreads; i++) {
    clientTCPStores.push_back(c10::make_intrusive<c10d::TCPStore>(
        "127.0.0.1", serverTCPStore->getPort(), numWorkers, false));
    clientStores.push_back(
        c10::make_intrusive<c10d::PrefixStore>(prefix, clientTCPStores[i]));
  }

  std::string expectedCounterRes =
      std::to_string(numThreads * numIterations + 1);

  for (auto i = 0; i < numThreads; i++) {
    threads.emplace_back(std::thread([&sem1,
                                      &sem2,
                                      &clientStores,
                                      i,
                                      &expectedCounterRes,
                                      &numIterations,
                                      &numThreads] {
      for (auto j = 0; j < numIterations; j++) {
        clientStores[i]->add("counter", 1);
      }
      // Let each thread set and get key on its client store
      std::string key = "thread_" + std::to_string(i);
      for (auto j = 0; j < numIterations; j++) {
        std::string val = "thread_val_" + std::to_string(j);
        c10d::test::set(*clientStores[i], key, val);
        c10d::test::check(*clientStores[i], key, val);
      }

      sem1.post();
      sem2.wait();
      // Check the counter results
      c10d::test::check(*clientStores[i], "counter", expectedCounterRes);
      // Now check other threads' written data
      for (auto j = 0; j < numThreads; j++) {
        if (j == i) {
          continue;
        }
        std::string key = "thread_" + std::to_string(i);
        std::string val = "thread_val_" + std::to_string(numIterations - 1);
        c10d::test::check(*clientStores[i], key, val);
      }
    }));
  }

  sem1.wait(numThreads);
  sem2.post(numThreads);

  for (auto& thread : threads) {
    thread.join();
  }

  serverThread.join();

  // Clear the store to test that client disconnect won't shutdown the store
  clientStores.clear();
  clientTCPStores.clear();

  // Check that the counter has the expected value
  c10d::test::check(*serverStore, "counter", expectedCounterRes);

  // Check that each threads' written data from the main thread
  for (auto i = 0; i < numThreads; i++) {
    std::string key = "thread_" + std::to_string(i);
    std::string val = "thread_val_" + std::to_string(numIterations - 1);
    c10d::test::check(*serverStore, key, val);
  }
}

void testWatchKeyCallback(const std::string& prefix = "") {
  // Callback function increments counter of the total number of callbacks that
  // were run
  int numCallbacksExecuted = 0;
  std::function<void(c10::optional<std::string>, c10::optional<std::string>)>
      callback = [&numCallbacksExecuted](
                     c10::optional<std::string> oldValue,
                     c10::optional<std::string> newValue) {
        std::ignore = oldValue;
        std::ignore = newValue;
        numCallbacksExecuted++;
      };

  const auto numThreads = 16;
  const auto numWorkers = numThreads + 1;
  auto serverTCPStore = _createServer(numWorkers);
  auto serverStore =
      c10::make_intrusive<c10d::PrefixStore>(prefix, serverTCPStore);

  // Start watching key
  std::string internalKey = "internalKey";
  for (auto i = 0; i < numThreads; i++) {
    serverStore->watchKey(internalKey + std::to_string(i), callback);
    serverStore->watchKey(
        internalKey + "counter" + std::to_string(i), callback);
  }

  // Each thread will have a client store to send/recv data
  std::vector<c10::intrusive_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<c10::intrusive_ptr<c10d::PrefixStore>> clientStores;
  for (auto i = 0; i < numThreads; i++) {
    clientTCPStores.push_back(c10::make_intrusive<c10d::TCPStore>(
        "127.0.0.1", serverTCPStore->getPort(), numWorkers, false));
    clientStores.push_back(
        c10::make_intrusive<c10d::PrefixStore>(prefix, clientTCPStores[i]));
  }

  std::vector<std::thread> threads;
  std::atomic<int> keyChangeOperationCount{0};
  for (auto i = 0; i < numThreads; i++) {
    threads.emplace_back(
        std::thread([&clientStores, &internalKey, &keyChangeOperationCount, i] {
          // Let each thread set and get key on its client store
          std::string key = internalKey + std::to_string(i);
          std::string keyCounter = internalKey + "counter" + std::to_string(i);
          std::string val = "thread_val_" + std::to_string(i);
          // The set, compareSet, add methods count as key change operations
          c10d::test::set(*clientStores[i], key, val);
          c10d::test::compareSet(*clientStores[i], key, val, "newValue");
          clientStores[i]->add(keyCounter, i);
          keyChangeOperationCount += 3;
          c10d::test::check(*clientStores[i], key, "newValue");
          c10d::test::check(*clientStores[i], keyCounter, std::to_string(i));
        }));
  }

  // Ensures that internal_key has been "set" and "get"
  for (auto& thread : threads) {
    thread.join();
  }

  // Check number of callbacks executed equal to number of key change operations
  EXPECT_EQ(keyChangeOperationCount, numCallbacksExecuted);
}

TEST(TCPStoreTest, testHelper) {
  testHelper();
}

TEST(TCPStoreTest, testHelperPrefix) {
  testHelper("testPrefix");
}

TEST(TCPStoreTest, testWatchKeyCallback) {
  testWatchKeyCallback();
}

TEST(TCPStoreTest, testWatchKeyCallbackWithPrefix) {
  testWatchKeyCallback("testPrefix");
}

// Helper function to create a key on the store, watch it, and run the callback
void _setCallback(
    c10d::Store& store,
    std::string key,
    std::exception_ptr& eptr,
    const c10::optional<std::string>& expectedOldValue,
    const c10::optional<std::string>& expectedNewValue) {
  // Test the correctness of new_value and old_value
  std::function<void(c10::optional<std::string>, c10::optional<std::string>)>
      callback = [expectedOldValue, expectedNewValue, &eptr](
                     c10::optional<std::string> oldValue,
                     c10::optional<std::string> newValue) {
        try {
          EXPECT_EQ(
              expectedOldValue.value_or("NONE"), oldValue.value_or("NONE"));
          EXPECT_EQ(
              expectedNewValue.value_or("NONE"), newValue.value_or("NONE"));
        } catch (...) {
          eptr = std::current_exception();
        }
        cv.notify_one();
      };
  store.watchKey(key, callback);
}

void _waitFinish() {
  std::unique_lock<std::mutex> lk(cvMutex);
  cv.wait_for(
      lk, std::chrono::duration<int, std::milli>(kStoreCallbackTimeoutMillis));
}

TEST(TCPStoreTest, testKeyUpdate) {
  auto store = _createServer();

  std::exception_ptr eptr = nullptr;
  std::string key = "testEmptyKeyValue";
  c10d::test::set(*store, key, "");
  // set does not block so wait for key to be set first
  store->get(key);
  _setCallback(*store, key, eptr, "", "2");
  c10d::test::set(*store, key, "2");
  _waitFinish();
  if (eptr)
    std::rethrow_exception(eptr);

  key = "testRegularKeyValue";
  c10d::test::set(*store, key, "1");
  // set does not block so wait for key to be set first
  store->get(key);
  _setCallback(*store, key, eptr, "1", "2");
  c10d::test::set(*store, key, "2");
  _waitFinish();
  if (eptr)
    std::rethrow_exception(eptr);
}

TEST(TCPStoreTest, testKeyCreate) {
  auto store = _createServer();

  std::exception_ptr eptr = nullptr;
  std::string key = "testWatchKeyCreate";
  _setCallback(*store, key, eptr, c10::nullopt, "2");
  c10d::test::set(*store, key, "2");
  _waitFinish();
  if (eptr)
    std::rethrow_exception(eptr);
}

TEST(TCPStoreTest, testKeyDelete) {
  auto store = _createServer();

  std::exception_ptr eptr = nullptr;
  std::string key = "testWatchKeyDelete";
  c10d::test::set(*store, key, "1");
  _setCallback(*store, key, eptr, "1", c10::nullopt);
  store->deleteKey(key);
  _waitFinish();
  if (eptr)
    std::rethrow_exception(eptr);
}

TEST(TCPStoreTest, testCleanShutdown) {
  int numWorkers = 2;

  auto serverTCPStore = std::make_unique<c10d::TCPStore>(
      "127.0.0.1",
      0,
      numWorkers,
      true,
      std::chrono::seconds(defaultTimeout),
      /* wait */ false);
  c10d::test::set(*serverTCPStore, "key", "val");

  auto clientTCPStore = c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      serverTCPStore->getPort(),
      numWorkers,
      false,
      std::chrono::seconds(defaultTimeout),
      /* wait */ false);
  clientTCPStore->get("key");

  auto clientThread = std::thread([&clientTCPStore] {
    EXPECT_THROW(clientTCPStore->get("invalid_key"), std::runtime_error);
  });

  // start server shutdown during a client request
  serverTCPStore = nullptr;

  clientThread.join();
}
