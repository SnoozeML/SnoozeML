//! Asynchronous stream supervisor managing broker ingestion, micro-batching dispatch, and response publishing.

pub mod driver;
pub mod message;
pub mod mock_stream;

#[cfg(feature = "streams-kafka")]
pub mod kafka;
#[cfg(feature = "streams-rabbitmq")]
pub mod rabbitmq;
#[cfg(feature = "streams-redis")]
pub mod redis_stream;

pub use crate::config::{StreamBindingConfig, StreamWireFormat};
use crate::registry::ModelRegistry;
use crate::server::metrics::ServerMetrics;
use crate::stream::driver::{ConsumedMessage, StreamDriver};
use crate::stream::message::{StreamInferRequest, StreamInferResponse};
use crate::stream::mock_stream::MockStreamDriver;
use crate::types::InferenceError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{watch, Semaphore};

/// Global counter generating unique correlation IDs when absent from requests.
static CORRELATION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Central supervisor coordinating streaming background workers, dynamic micro-batching, and egress publishing.
pub struct StreamSupervisor {
    registry: ModelRegistry,
    metrics: Arc<ServerMetrics>,
    configs: Vec<StreamBindingConfig>,
    custom_drivers: std::sync::RwLock<std::collections::HashMap<String, Arc<dyn StreamDriver>>>,
}

impl StreamSupervisor {
    /// Creates a new `StreamSupervisor` with the specified registry, metrics collector, and stream configurations.
    #[must_use]
    pub fn new(
        registry: ModelRegistry,
        metrics: Arc<ServerMetrics>,
        configs: Vec<StreamBindingConfig>,
    ) -> Self {
        Self {
            registry,
            metrics,
            configs,
            custom_drivers: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Registers a custom or pre-configured [`StreamDriver`] for a specific stream name.
    pub fn register_driver(&self, stream_name: &str, driver: Arc<dyn StreamDriver>) {
        if let Ok(mut guard) = self.custom_drivers.write() {
            guard.insert(stream_name.to_string(), driver);
        }
    }

    /// Spawns all configured streaming workers as background Tokio tasks.
    ///
    /// Each stream worker runs completely asynchronously and non-blocking relative to HTTP and gRPC listeners.
    pub fn spawn_workers(self: Arc<Self>, shutdown_rx: watch::Receiver<()>) {
        for config in self.configs.clone() {
            let supervisor = self.clone();
            let worker_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                supervisor.run_stream_worker(config, worker_shutdown).await;
            });
        }
    }

    /// Factory instantiating the appropriate [`StreamDriver`] implementation based on configuration.
    async fn create_driver(
        &self,
        config: &StreamBindingConfig,
    ) -> Result<Arc<dyn StreamDriver>, InferenceError> {
        // Check custom drivers first (e.g. for testing)
        if let Ok(guard) = self.custom_drivers.read() {
            if let Some(driver) = guard.get(&config.name) {
                return Ok(driver.clone());
            }
        }

        match config.broker.to_ascii_lowercase().as_str() {
            "mock" | "memory" => {
                let driver = Arc::new(MockStreamDriver::new(&config.input_topic));
                Ok(driver)
            }
            #[cfg(feature = "streams-rabbitmq")]
            "rabbitmq" | "amqp" => {
                let prefetch = u16::try_from(config.concurrency).unwrap_or(32);
                let driver = rabbitmq::RabbitMqDriver::connect(
                    &config.endpoint,
                    &config.input_topic,
                    prefetch,
                )
                .await?;
                Ok(Arc::new(driver))
            }
            #[cfg(not(feature = "streams-rabbitmq"))]
            "rabbitmq" | "amqp" => Err(InferenceError::Runtime(
                "RabbitMQ streaming support is disabled. Recompile SnoozeML with feature flag 'streams-rabbitmq'".into(),
            )),
            #[cfg(feature = "streams-redis")]
            "redis" => {
                let group = config.consumer_group.as_deref().unwrap_or("snoozeml-group");
                let consumer_name = format!("snoozeml-worker-{}", std::process::id());
                let driver = redis_stream::RedisStreamDriver::connect(
                    &config.endpoint,
                    &config.input_topic,
                    group,
                    consumer_name,
                )
                .await?;
                Ok(Arc::new(driver))
            }
            #[cfg(not(feature = "streams-redis"))]
            "redis" => Err(InferenceError::Runtime(
                "Redis Streams support is disabled. Recompile SnoozeML with feature flag 'streams-redis'".into(),
            )),
            #[cfg(feature = "streams-kafka")]
            "kafka" | "redpanda" => {
                let driver = kafka::KafkaStreamDriver::connect(
                    &config.endpoint,
                    &config.input_topic,
                )
                .await?;
                Ok(Arc::new(driver))
            }
            #[cfg(not(feature = "streams-kafka"))]
            "kafka" | "redpanda" => Err(InferenceError::Runtime(
                "Kafka streaming support is disabled. Recompile SnoozeML with feature flag 'streams-kafka'".into(),
            )),
            unknown => Err(InferenceError::InvalidInput(format!(
                "Unsupported stream broker '{unknown}'. Supported brokers: 'kafka', 'rabbitmq', 'redis', 'mock'"
            ))),
        }
    }

    /// Dedicated async loop managing an individual stream worker.
    async fn run_stream_worker(
        self: Arc<Self>,
        config: StreamBindingConfig,
        mut shutdown_rx: watch::Receiver<()>,
    ) {
        tracing::info!(
            stream = %config.name,
            broker = %config.broker,
            model_id = ?config.model_id,
            input_topic = %config.input_topic,
            output_topic = ?config.output_topic,
            auto_create = config.auto_create_topics,
            concurrency = config.concurrency,
            "Initializing stream worker"
        );

        let driver = match self.create_driver(&config).await {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(
                    stream = %config.name,
                    error = %e,
                    "Failed to initialize stream driver; worker terminated"
                );
                return;
            }
        };

        // Automatic topic / queue / stream provisioning if enabled
        if config.auto_create_topics {
            tracing::info!(
                stream = %config.name,
                input_topic = %config.input_topic,
                output_topic = ?config.output_topic,
                "Ensuring topics/queues exist on broker"
            );

            if let Err(e) = driver
                .ensure_topic_exists(&config.input_topic, config.partitions)
                .await
            {
                tracing::warn!(stream = %config.name, topic = %config.input_topic, error = %e, "Failed auto-creating input topic");
            }
            if let Some(ref out_topic) = config.output_topic {
                if let Err(e) = driver
                    .ensure_topic_exists(out_topic, config.partitions)
                    .await
                {
                    tracing::warn!(stream = %config.name, topic = %out_topic, error = %e, "Failed auto-creating output topic");
                }
            }
            if let Some(ref err_topic) = config.error_topic {
                if let Err(e) = driver
                    .ensure_topic_exists(err_topic, config.partitions)
                    .await
                {
                    tracing::warn!(stream = %config.name, topic = %err_topic, error = %e, "Failed auto-creating error topic");
                }
            }
        }

        // Bounded concurrency semaphore for backpressure
        let semaphore = Arc::new(Semaphore::new(config.concurrency));

        tracing::info!(stream = %config.name, "Stream worker successfully started; consuming messages");

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    tracing::info!(stream = %config.name, "Stream worker received shutdown signal; draining in-flight requests");
                    break;
                }
                poll_res = driver.poll_next(Duration::from_millis(150)) => {
                    match poll_res {
                        Ok(Some(msg)) => {
                            let permit = match semaphore.clone().acquire_owned().await {
                                Ok(p) => p,
                                Err(_) => break, // Semaphore closed during shutdown
                            };

                            let supervisor = self.clone();
                            let driver_clone = driver.clone();
                            let config_clone = config.clone();

                            tokio::spawn(async move {
                                supervisor.process_single_message(msg, driver_clone, config_clone).await;
                                drop(permit);
                            });
                        }
                        Ok(None) => {
                            // Idle polling window; yields control to Tokio scheduler
                        }
                        Err(e) => {
                            tracing::warn!(
                                stream = %config.name,
                                error = %e,
                                "Transient error polling stream broker; backing off"
                            );
                            tokio::time::sleep(Duration::from_millis(500)).await;
                        }
                    }
                }
            }
        }

        // Graceful drain: wait up to 5 seconds for in-flight tasks to release semaphore permits
        let _ = tokio::time::timeout(
            Duration::from_secs(5),
            semaphore.acquire_many(u32::try_from(config.concurrency).unwrap_or(1)),
        )
        .await;

        tracing::info!(stream = %config.name, "Stream worker successfully stopped");
    }

    /// Processes an individual stream message through model inference and response publication.
    async fn process_single_message(
        &self,
        msg: ConsumedMessage,
        driver: Arc<dyn StreamDriver>,
        config: StreamBindingConfig,
    ) {
        let start = Instant::now();
        self.metrics
            .record_stream_consumed(driver.broker_name(), &config.input_topic);

        // 1. Decode payload
        let req = match StreamInferRequest::from_bytes(&msg.payload, config.format) {
            Ok(r) => r,
            Err(e) => {
                let err_str = e.to_string();
                tracing::warn!(stream = %config.name, error = %err_str, "Received malformed stream request payload");
                self.metrics.record_stream_error(
                    driver.broker_name(),
                    &config.input_topic,
                    "malformed_request",
                );

                let corr_id = msg
                    .headers
                    .get("correlation_id")
                    .cloned()
                    .unwrap_or_else(Self::generate_correlation_id);

                let default_model_name = config.model_id.as_deref().unwrap_or("unknown");
                let err_resp = StreamInferResponse::error(
                    corr_id,
                    default_model_name,
                    err_str,
                    start.elapsed().as_secs_f64() * 1000.0,
                );

                let target_topic = config
                    .error_topic
                    .as_deref()
                    .or(config.output_topic.as_deref());
                if let Some(topic) = target_topic {
                    if let Ok(bytes) = err_resp.to_bytes(config.format) {
                        let _ = driver.publish(topic, &bytes).await;
                    }
                }
                let _ = msg.acknowledger.ack().await;
                return;
            }
        };

        let corr_id = req
            .id
            .clone()
            .or_else(|| msg.headers.get("correlation_id").cloned())
            .unwrap_or_else(Self::generate_correlation_id);

        let target_model_id = match req.model_id.clone().or_else(|| config.model_id.clone()) {
            Some(id) if !id.trim().is_empty() => id,
            _ => {
                let err_str = "No model_id specified in stream request and no default model_id configured for stream binding".to_string();
                tracing::warn!(stream = %config.name, "Rejecting stream request with missing model_id");
                self.metrics.record_stream_error(
                    driver.broker_name(),
                    &config.input_topic,
                    "missing_model_id",
                );

                let err_resp = StreamInferResponse::error(
                    corr_id,
                    "unknown",
                    err_str,
                    start.elapsed().as_secs_f64() * 1000.0,
                );

                let target_topic = req
                    .reply_to
                    .as_deref()
                    .or(config.error_topic.as_deref())
                    .or(config.output_topic.as_deref());
                if let Some(topic) = target_topic {
                    if let Ok(bytes) = err_resp.to_bytes(config.format) {
                        let _ = driver.publish(topic, &bytes).await;
                    }
                }
                let _ = msg.acknowledger.ack().await;
                return;
            }
        };
        let reply_to = req.reply_to.clone();

        // 2. Extract tensors
        let payloads = match req.extract_payloads() {
            Ok(p) => p,
            Err(e) => {
                let err_str = e.to_string();
                tracing::warn!(stream = %config.name, model_id = %target_model_id, error = %err_str, "Invalid tensor payloads");
                self.metrics.record_stream_error(
                    driver.broker_name(),
                    &config.input_topic,
                    "invalid_tensors",
                );

                let err_resp = StreamInferResponse::error(
                    corr_id,
                    target_model_id,
                    err_str,
                    start.elapsed().as_secs_f64() * 1000.0,
                );

                let target_topic = reply_to
                    .as_deref()
                    .or(config.error_topic.as_deref())
                    .or(config.output_topic.as_deref());
                if let Some(topic) = target_topic {
                    if let Ok(bytes) = err_resp.to_bytes(config.format) {
                        let _ = driver.publish(topic, &bytes).await;
                    }
                }
                let _ = msg.acknowledger.ack().await;
                return;
            }
        };

        // 3. Route to ModelRegistry & dispatch inference
        let infer_res = match self.registry.route_for_inference(&target_model_id).await {
            Ok(controller) => controller.infer(payloads).await,
            Err(e) => Err(e),
        };

        let elapsed = start.elapsed();
        let elapsed_us = elapsed.as_micros() as u64;
        let duration_ms = (elapsed_us as f64) / 1000.0;

        // 4. Formulate response and publish
        match infer_res {
            Ok(outputs) => {
                self.metrics.record_inference(true, elapsed_us);
                let target_topic = reply_to.as_deref().or(config.output_topic.as_deref());

                if let Some(topic) = target_topic {
                    self.metrics
                        .record_stream_published(driver.broker_name(), topic);

                    let resp = StreamInferResponse::success(
                        corr_id,
                        target_model_id,
                        outputs,
                        duration_ms,
                    );

                    if let Ok(bytes) = resp.to_bytes(config.format) {
                        if let Err(e) = driver.publish(topic, &bytes).await {
                            tracing::error!(stream = %config.name, topic = %topic, error = %e, "Failed to publish stream response");
                            self.metrics.record_stream_error(
                                driver.broker_name(),
                                topic,
                                "publish_failed",
                            );
                            let _ = msg.acknowledger.nack(true).await;
                            return;
                        }
                    }
                }
                let _ = msg.acknowledger.ack().await;
            }
            Err(e) => {
                self.metrics.record_inference(false, elapsed_us);
                let err_str = e.to_string();
                tracing::warn!(stream = %config.name, model_id = %target_model_id, error = %err_str, "Inference execution failed on stream job");
                self.metrics.record_stream_error(
                    driver.broker_name(),
                    &config.input_topic,
                    "inference_error",
                );

                let err_resp =
                    StreamInferResponse::error(corr_id, target_model_id, err_str, duration_ms);
                let target_topic = config
                    .error_topic
                    .as_deref()
                    .or(reply_to.as_deref())
                    .or(config.output_topic.as_deref());

                if let Some(topic) = target_topic {
                    if let Ok(bytes) = err_resp.to_bytes(config.format) {
                        let _ = driver.publish(topic, &bytes).await;
                    }
                }
                let _ = msg.acknowledger.ack().await;
            }
        }
    }

    /// Generates a sequential fallback correlation identifier.
    #[must_use]
    pub fn generate_correlation_id() -> String {
        let count = CORRELATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("snooze-stream-{count}")
    }
}
