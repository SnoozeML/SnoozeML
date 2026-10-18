//! Dynamic micro-batching engine for accumulating asynchronous requests,
//! concatenating multi-dimensional tensors along Axis(0), and distributing sliced outputs.

use crate::types::{InferenceError, InferenceJob, TensorPayload};
use ndarray::{concatenate, Axis, Slice};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

/// Configuration parameters for the dynamic batch accumulator.
#[derive(Debug, Clone)]
pub struct DynamicBatcherConfig {
    /// Maximum aggregate batch size before immediately dispatching to execution.
    pub max_batch_size: usize,
    /// Maximum duration to wait for additional requests before flushing a partial batch.
    pub max_delay: Duration,
    /// Capacity of the internal MPSC submission queue.
    pub queue_capacity: usize,
}

impl Default for DynamicBatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            max_delay: Duration::from_millis(10),
            queue_capacity: 1024,
        }
    }
}

/// Tuple representing an awaiting client callback channel and its original input batch size.
pub type BatchResponder = (
    oneshot::Sender<Result<Vec<TensorPayload>, InferenceError>>,
    usize,
);

/// Dynamic micro-batching engine wrapping a Tokio MPSC submission channel.
#[derive(Debug, Clone)]
pub struct DynamicBatcher {
    tx: mpsc::Sender<InferenceJob>,
}

impl DynamicBatcher {
    /// Spawns a new dynamic batch accumulator background task.
    ///
    /// # Panics
    /// None. Execution errors are safely forwarded to job response channels.
    pub fn spawn<F>(config: DynamicBatcherConfig, run_batch_fn: F) -> Self
    where
        F: Fn(Vec<TensorPayload>, Vec<BatchResponder>) + Send + Sync + 'static,
    {
        let (tx, mut rx) = mpsc::channel::<InferenceJob>(config.queue_capacity);
        let run_fn = Arc::new(run_batch_fn);

        tokio::spawn(async move {
            let mut queue: Vec<InferenceJob> = Vec::with_capacity(config.max_batch_size);
            let mut current_accumulated_batch = 0usize;
            let sleep = tokio::time::sleep(config.max_delay);
            tokio::pin!(sleep);
            let mut timer_active = false;

            loop {
                if queue.is_empty() {
                    timer_active = false;
                    // Drain up to max_batch_size available items atomically from channel
                    let count = rx.recv_many(&mut queue, config.max_batch_size).await;
                    if count == 0 {
                        break; // Channel closed
                    }

                    for job in queue.as_slice() {
                        let job_batch = job
                            .inputs
                            .first()
                            .map(|t| t.batch_size())
                            .unwrap_or(1)
                            .max(1);
                        current_accumulated_batch =
                            current_accumulated_batch.saturating_add(job_batch);
                    }

                    if current_accumulated_batch >= config.max_batch_size {
                        Self::flush(&mut queue, &run_fn);
                        current_accumulated_batch = 0;
                    } else {
                        // Start deadline countdown for this partial batch
                        sleep.as_mut().reset(Instant::now() + config.max_delay);
                        timer_active = true;
                    }
                } else {
                    let needed = config
                        .max_batch_size
                        .saturating_sub(current_accumulated_batch)
                        .max(1);
                    let prev_len = queue.len();

                    tokio::select! {
                        count = rx.recv_many(&mut queue, needed) => {
                            if count == 0 {
                                // Channel closed; flush remaining items and terminate
                                Self::flush(&mut queue, &run_fn);
                                break;
                            }

                            if let Some(new_jobs) = queue.get(prev_len..) {
                                for j in new_jobs {
                                    let j_batch = j.inputs.first().map(|t| t.batch_size()).unwrap_or(1).max(1);
                                    current_accumulated_batch = current_accumulated_batch.saturating_add(j_batch);
                                }
                            }

                            if current_accumulated_batch >= config.max_batch_size {
                                Self::flush(&mut queue, &run_fn);
                                current_accumulated_batch = 0;
                                timer_active = false;
                            }
                        }
                        _ = &mut sleep, if timer_active => {
                            Self::flush(&mut queue, &run_fn);
                            current_accumulated_batch = 0;
                            timer_active = false;
                        }
                    }
                }
            }
        });

        Self { tx }
    }

    /// Submits an inference job into the micro-batch accumulator.
    ///
    /// # Errors
    /// Returns [`InferenceError::QueueError`] if the batcher channel is full or offline.
    pub async fn submit(&self, job: InferenceJob) -> Result<(), InferenceError> {
        self.tx
            .send(job)
            .await
            .map_err(|e| InferenceError::QueueError(format!("Batch accumulator unavailable: {e}")))
    }

    /// Submits an inference job into the micro-batch accumulator immediately, failing fast if full.
    ///
    /// # Errors
    /// Returns [`InferenceError::QueueError`] if the batcher channel is full or offline.
    pub fn try_submit(&self, job: InferenceJob) -> Result<(), InferenceError> {
        self.tx.try_send(job).map_err(|e| {
            InferenceError::QueueError(format!("Batch accumulator unavailable or full: {e}"))
        })
    }

    /// Returns the number of jobs currently waiting in the batcher queue channel.
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.tx.max_capacity().saturating_sub(self.tx.capacity())
    }

    /// Returns the maximum capacity of the batcher queue channel.
    #[must_use]
    pub fn queue_capacity(&self) -> usize {
        self.tx.max_capacity()
    }

    /// Internal collation and dispatch logic when queue threshold or timer expires.
    fn flush<F>(queue: &mut Vec<InferenceJob>, run_fn: &Arc<F>)
    where
        F: Fn(Vec<TensorPayload>, Vec<BatchResponder>) + Send + Sync + 'static,
    {
        if queue.is_empty() {
            return;
        }

        let mut jobs = std::mem::replace(queue, Vec::with_capacity(queue.capacity()));

        // Filter out abandoned / canceled jobs whose receivers have dropped
        jobs.retain(|job| !job.response_tx.is_closed());
        if jobs.is_empty() {
            return;
        }

        // Fast path: single job accumulated. Forward directly without collation or float copies.
        if jobs.len() == 1 {
            if let Some(job) = jobs.pop() {
                let slice_size = job
                    .inputs
                    .first()
                    .map(|t| t.batch_size())
                    .unwrap_or(1)
                    .max(1);
                let responders = vec![(job.response_tx, slice_size)];
                run_fn(job.inputs, responders);
                return;
            }
        }

        let mut responders = Vec::with_capacity(jobs.len());
        let mut unbatched_requests: Vec<Vec<TensorPayload>> = Vec::with_capacity(jobs.len());

        for job in jobs {
            let slice_size = job
                .inputs
                .first()
                .map(|t| t.batch_size())
                .unwrap_or(1)
                .max(1);
            responders.push((job.response_tx, slice_size));
            unbatched_requests.push(job.inputs);
        }

        // Collate inputs along Axis(0)
        let collated = Self::collate_inputs(&unbatched_requests);

        match collated {
            Ok(batched_inputs) => {
                run_fn(batched_inputs, responders);
            }
            Err(err) => {
                // Shape or name mismatch during collation: notify all awaiting callers
                for (tx, _) in responders {
                    let _ = tx.send(Err(err.clone()));
                }
            }
        }
    }

    /// Collates multiple requests into batched tensors concatenated along Axis 0.
    ///
    /// # Errors
    /// Returns [`InferenceError::ShapeMismatch`] or [`InferenceError::InvalidInput`]
    /// if input counts, names, or trailing dimensions do not match across jobs,
    /// or if any input tensor is 0-D (scalar).
    pub fn collate_inputs(
        unbatched: &[Vec<TensorPayload>],
    ) -> Result<Vec<TensorPayload>, InferenceError> {
        let first_req = unbatched
            .first()
            .ok_or_else(|| InferenceError::InvalidInput("Cannot collate empty job list".into()))?;

        let num_tensors = first_req.len();
        if num_tensors == 0 {
            return Err(InferenceError::InvalidInput(
                "Request has 0 input tensors".into(),
            ));
        }

        // Validate that no request contains 0-D (scalar) tensors, which cannot be concatenated along Axis(0)
        for (req_idx, req) in unbatched.iter().enumerate() {
            for tensor in req {
                if tensor.shape.is_empty() {
                    return Err(InferenceError::InvalidInput(format!(
                        "Cannot collate 0-D (scalar) tensor '{}' in request {req_idx} along Axis(0)",
                        tensor.name
                    )));
                }
            }
        }

        // Fast path: single job requires no collation or memory reallocation
        if unbatched.len() == 1 {
            return Ok(first_req.clone());
        }

        let mut batched_inputs = Vec::with_capacity(num_tensors);

        for tensor_idx in 0..num_tensors {
            let sample_tensor = first_req.get(tensor_idx).ok_or_else(|| {
                InferenceError::InvalidInput(format!("Missing tensor index {tensor_idx}"))
            })?;

            let tensor_name = &sample_tensor.name;
            let expected_trailing = sample_tensor.trailing_shape();

            let mut views = Vec::with_capacity(unbatched.len());

            for (req_idx, req) in unbatched.iter().enumerate() {
                let tensor = req.get(tensor_idx).ok_or_else(|| {
                    InferenceError::ShapeMismatch(format!(
                        "Request {req_idx} missing tensor at index {tensor_idx}"
                    ))
                })?;

                if &tensor.name != tensor_name {
                    return Err(InferenceError::ShapeMismatch(format!(
                        "Request {req_idx} tensor name mismatch: expected '{tensor_name}', got '{}'",
                        tensor.name
                    )));
                }

                if tensor.trailing_shape() != expected_trailing {
                    return Err(InferenceError::ShapeMismatch(format!(
                        "Request {req_idx} trailing shape mismatch for tensor '{tensor_name}': expected {expected_trailing:?}, got {:?}",
                        tensor.trailing_shape()
                    )));
                }

                views.push(tensor.data.view());
            }

            let concatenated = concatenate(Axis(0), &views).map_err(|e| {
                InferenceError::ShapeMismatch(format!(
                    "Failed to concatenate tensor '{tensor_name}' along Axis(0): {e}"
                ))
            })?;

            batched_inputs.push(TensorPayload {
                name: tensor_name.clone(),
                shape: concatenated.shape().to_vec(),
                data: concatenated,
            });
        }

        Ok(batched_inputs)
    }

    /// Splits batched outputs along Axis 0 according to each client's original input slice size,
    /// optionally validating the expected number of output tensors.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] if:
    /// - The backend returns 0 output tensors.
    /// - The backend output count does not match `expected_tensor_count` (if provided).
    /// - Total output batch size does not match expected sum of `slice_sizes`.
    pub fn split_outputs_with_expected(
        batched_outputs: &[TensorPayload],
        slice_sizes: &[usize],
        expected_tensor_count: Option<usize>,
    ) -> Result<Vec<Vec<TensorPayload>>, InferenceError> {
        if batched_outputs.is_empty() {
            return Err(InferenceError::Runtime(
                "Backend returned 0 output tensors".into(),
            ));
        }

        if let Some(expected_count) = expected_tensor_count {
            if batched_outputs.len() != expected_count {
                return Err(InferenceError::Runtime(format!(
                    "Backend returned {} output tensors, expected {expected_count}",
                    batched_outputs.len()
                )));
            }
        }

        let total_expected_batch: usize = slice_sizes.iter().sum();

        // Fast path: single client requesting the full batch avoids slicing and copying
        if slice_sizes.len() == 1 {
            let expected = slice_sizes.first().copied().unwrap_or(0);
            for output_tensor in batched_outputs {
                let output_batch = output_tensor.batch_size();
                if output_batch != expected {
                    return Err(InferenceError::Runtime(format!(
                        "Output tensor '{}' batch size {output_batch} does not match expected total {expected}",
                        output_tensor.name
                    )));
                }
            }
            return Ok(vec![batched_outputs.to_vec()]);
        }

        let mut per_client_outputs: Vec<Vec<TensorPayload>> = (0..slice_sizes.len())
            .map(|_| Vec::with_capacity(batched_outputs.len()))
            .collect();

        for output_tensor in batched_outputs {
            let output_batch = output_tensor.batch_size();
            if output_batch != total_expected_batch {
                return Err(InferenceError::Runtime(format!(
                    "Output tensor '{}' batch size {output_batch} does not match expected total {total_expected_batch}",
                    output_tensor.name
                )));
            }

            let mut offset = 0usize;
            for (client_idx, &slice_len) in slice_sizes.iter().enumerate() {
                let end = offset.saturating_add(slice_len);
                let slice_view = output_tensor
                    .data
                    .slice_axis(Axis(0), Slice::from(offset..end))
                    .to_owned();

                let client_tensor = TensorPayload {
                    name: output_tensor.name.clone(),
                    shape: slice_view.shape().to_vec(),
                    data: slice_view,
                };

                let client_vec = per_client_outputs.get_mut(client_idx).ok_or_else(|| {
                    InferenceError::Runtime(format!("Client index {client_idx} out of bounds"))
                })?;
                client_vec.push(client_tensor);

                offset = end;
            }
        }

        Ok(per_client_outputs)
    }

    /// Splits batched outputs along Axis 0 according to each client's original input slice size.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] if:
    /// - The backend returns 0 output tensors.
    /// - Total output batch size does not match expected sum of `slice_sizes`.
    pub fn split_outputs(
        batched_outputs: &[TensorPayload],
        slice_sizes: &[usize],
    ) -> Result<Vec<Vec<TensorPayload>>, InferenceError> {
        Self::split_outputs_with_expected(batched_outputs, slice_sizes, None)
    }
}
