// src/calculations.rs - High-performance financial calculations for massive datasets
use nalgebra::DVector;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use futures::future;
use chrono::NaiveDate;

/// High-performance financial calculator optimized for massive datasets
pub struct SIMDFinancialCalculator;

impl SIMDFinancialCalculator {
    /// Calculate daily returns optimized - zero extra allocations
    #[inline(always)]
    pub fn daily_returns_simd(prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < 2 {
            return Vec::new();
        }

        let mut returns = vec![0.0; n - 1];
        
        // For massive datasets, use parallel processing
        if n > 10_000 {
            returns.par_iter_mut().enumerate().for_each(|(i, ret)| {
                let prev = prices[i];
                let curr = prices[i + 1];
                *ret = if prev != 0.0 { (curr - prev) / prev } else { 0.0 };
            });
        } else {
            // Scalar version for smaller datasets
            for i in 1..n {
                let prev = prices[i - 1];
                let curr = prices[i];
                returns[i - 1] = if prev != 0.0 { (curr - prev) / prev } else { 0.0 };
            }
        }

        returns
    }

    /// Lightning-fast volatility calculation - one-pass algorithm
    #[inline(always)]
    pub fn volatility_optimized(returns: &[f64]) -> f64 {
        let n = returns.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        // One-pass parallel variance calculation
        let (sum, sumsq) = if returns.len() > 50_000 {
            returns.par_chunks(1 << 15).map(|chunk| {
                let mut s = 0.0;
                let mut ss = 0.0;
                for &x in chunk {
                    s += x;
                    ss += x * x;
                }
                (s, ss)
            }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            let mut s = 0.0;
            let mut ss = 0.0;
            for &x in returns {
                s += x;
                ss += x * x;
            }
            (s, ss)
        };

        let mean = sum / n;
        let variance = (sumsq / n) - mean * mean;
        variance.max(0.0).sqrt() * 252.0f64.sqrt()
    }

    /// Ultra-fast correlation with SIMD operations
    #[inline(always)]
    pub fn correlation_simd(returns1: &[f64], returns2: &[f64]) -> f64 {
        if returns1.len() != returns2.len() || returns1.is_empty() {
            return 0.0;
        }

        let n = returns1.len();

        // For massive datasets, use parallel processing
        if n > 50_000 {
            return Self::correlation_parallel(returns1, returns2);
        }

        // For medium datasets, use SIMD
        Self::correlation_simd_vectorized(returns1, returns2)
    }

    /// Parallel correlation for massive datasets
    fn correlation_parallel(returns1: &[f64], returns2: &[f64]) -> f64 {
        let n = returns1.len() as f64;

        // Parallel computation of sums
        let (sum1, sum2): (f64, f64) =
            rayon::join(|| returns1.par_iter().sum(), || returns2.par_iter().sum());

        let (sum1_sq, sum2_sq_and_prod) = rayon::join(
            || returns1.par_iter().map(|&x| x * x).sum::<f64>(),
            || {
                let sum2_sq: f64 = returns2.par_iter().map(|&x| x * x).sum();
                let sum_prod: f64 = returns1
                    .par_iter()
                    .zip(returns2.par_iter())
                    .map(|(&x, &y)| x * y)
                    .sum();
                (sum2_sq, sum_prod)
            },
        );
        let (sum2_sq_val, sum_prod) = sum2_sq_and_prod;

        let numerator: f64 = n * sum_prod - sum1 * sum2;
        let denominator: f64 = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq_val - sum2 * sum2)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// SIMD-vectorized correlation calculation
    fn correlation_simd_vectorized(returns1: &[f64], returns2: &[f64]) -> f64 {
        let n = returns1.len() as f64;
        
        let sum1: f64 = returns1.iter().sum();
        let sum2: f64 = returns2.iter().sum();
        let sum1_sq: f64 = returns1.iter().map(|x| x * x).sum();
        let sum2_sq: f64 = returns2.iter().map(|x| x * x).sum();
        let sum_prod: f64 = returns1.iter().zip(returns2.iter()).map(|(x, y)| x * y).sum();

        let numerator = n * sum_prod - sum1 * sum2;
        let denominator = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Tracking error without building excess_returns vector
    #[inline(always)]
    pub fn tracking_error_optimized(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
        if portfolio_returns.len() != benchmark_returns.len() || portfolio_returns.is_empty() {
            return 0.0;
        }

        let n = portfolio_returns.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        // Calculate variance of excess returns directly without temp vector
        let (sum, sumsq) = if portfolio_returns.len() > 50_000 {
            portfolio_returns.par_iter().zip(benchmark_returns.par_iter()).map(|(&p, &b)| {
                let excess = p - b;
                (excess, excess * excess)
            }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            let mut s = 0.0;
            let mut ss = 0.0;
            for (&p, &b) in portfolio_returns.iter().zip(benchmark_returns.iter()) {
                let excess = p - b;
                s += excess;
                ss += excess * excess;
            }
            (s, ss)
        };

        let mean = sum / n;
        let variance = (sumsq / n) - mean * mean;
        variance.max(0.0).sqrt() * 252.0f64.sqrt()
    }

    /// Cumulative return calculation - handles massive time series
    #[inline(always)]
    pub fn cumulative_return(prices: &[(NaiveDate, f64)]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let start_price = prices[0].1;
        let end_price = prices[prices.len() - 1].1;

        if start_price != 0.0 {
            (end_price - start_price) / start_price
        } else {
            0.0
        }
    }

    /// Cumulative return from price vector (for handlers)
    #[inline(always)]
    pub fn cumulative_return_from_prices(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let start_price = prices[0];
        let end_price = prices[prices.len() - 1];

        if start_price != 0.0 {
            (end_price - start_price) / start_price
        } else {
            0.0
        }
    }

    /// Memory-efficient streaming calculations for massive datasets
    pub async fn calculate_volatility_streaming<F>(data_source: F, chunk_size: usize) -> f64
    where
        F: Fn(usize, usize) -> futures::future::BoxFuture<'static, Vec<f64>>,
    {
        let mut rolling_sum = 0.0;
        let mut rolling_sum_squares = 0.0;
        let mut total_count = 0usize;
        let mut offset = 0usize;

        loop {
            let chunk = data_source(offset, chunk_size).await;
            if chunk.is_empty() {
                break;
            }

            // Process chunk with SIMD
            for &value in &chunk {
                rolling_sum += value;
                rolling_sum_squares += value * value;
                total_count += 1;
            }

            offset += chunk.len();
        }

        if total_count < 2 {
            return 0.0;
        }

        let mean = rolling_sum / total_count as f64;
        let variance = (rolling_sum_squares / total_count as f64) - (mean * mean);
        variance.sqrt() * (252.0_f64).sqrt()
    }
}

/// Batch processing utilities for massive datasets
pub struct BatchProcessor;

impl BatchProcessor {
    /// Process massive datasets in chunks to avoid memory issues
    pub async fn process_large_dataset<T, F, Fut>(
        data: Vec<T>,
        chunk_size: usize,
        processor: F,
    ) -> Vec<f64>
    where
        T: Send + Clone + 'static,
        F: Fn(Vec<T>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<f64>> + Send + 'static,
    {
        let processor = Arc::new(processor);
        let mut tasks = Vec::new();

        for chunk in data.chunks(chunk_size) {
            let chunk_data = chunk.to_vec();
            let proc = processor.clone();

            tasks.push(tokio::spawn(async move { proc(chunk_data).await }));
        }

        let results = futures::future::join_all(tasks).await;
        let mut all_results = Vec::new();

        for result in results {
            if let Ok(chunk_results) = result {
                all_results.extend(chunk_results);
            }
        }

        all_results
    }

    /// Parallel correlation matrix calculation for multiple portfolios
    pub async fn correlation_matrix_parallel(
        portfolio_data: HashMap<String, Vec<f64>>,
    ) -> HashMap<(String, String), f64> {
        let portfolios: Vec<String> = portfolio_data.keys().cloned().collect();
        let mut correlations = HashMap::new();

        // Generate all portfolio pairs
        let mut pairs = Vec::new();
        for i in 0..portfolios.len() {
            for j in (i + 1)..portfolios.len() {
                pairs.push((portfolios[i].clone(), portfolios[j].clone()));
            }
        }

        // Calculate correlations in parallel
        let correlation_results: Vec<_> = pairs
            .par_iter()
            .map(|(p1, p2)| {
                let returns1 = &portfolio_data[p1];
                let returns2 = &portfolio_data[p2];
                let correlation = SIMDFinancialCalculator::correlation_simd(returns1, returns2);
                ((p1.clone(), p2.clone()), correlation)
            })
            .collect();

        for (pair, correlation) in correlation_results {
            correlations.insert(pair, correlation);
        }

        correlations
    }
}

/// Rolling statistics for real-time updates on streaming data
#[derive(Clone, Debug)]
pub struct StreamingStatistics {
    window_size: usize,
    values: std::collections::VecDeque<f64>,
    sum: f64,
    sum_squares: f64,
    min_value: f64,
    max_value: f64,
    last_return: Option<f64>,
}

impl StreamingStatistics {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: std::collections::VecDeque::with_capacity(window_size),
            sum: 0.0,
            sum_squares: 0.0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            last_return: None,
        }
    }

    /// Add new value and get updated statistics in O(1) time
    pub fn update(&mut self, value: f64) -> StreamingMetrics {
        // Calculate return if we have previous value
        let return_value = if let Some(last) = self.last_return {
            Some((value - last) / last)
        } else {
            None
        };

        if let Some(ret) = return_value {
            // Remove oldest value if at capacity
            if self.values.len() >= self.window_size {
                if let Some(old_value) = self.values.pop_front() {
                    self.sum -= old_value;
                    self.sum_squares -= old_value * old_value;
                }
            }

            // Add new return
            self.values.push_back(ret);
            self.sum += ret;
            self.sum_squares += ret * ret;

            // Update min/max
            self.min_value = self.min_value.min(ret);
            self.max_value = self.max_value.max(ret);
        }

        self.last_return = Some(value);

        StreamingMetrics {
            count: self.values.len(),
            mean: if self.values.is_empty() {
                0.0
            } else {
                self.sum / self.values.len() as f64
            },
            volatility: self.calculate_volatility(),
            min_return: if self.min_value == f64::INFINITY {
                0.0
            } else {
                self.min_value
            },
            max_return: if self.max_value == f64::NEG_INFINITY {
                0.0
            } else {
                self.max_value
            },
            latest_return: return_value.unwrap_or(0.0),
        }
    }

    #[inline(always)]
    fn calculate_volatility(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }

        let n = self.values.len() as f64;
        let mean = self.sum / n;
        let variance = (self.sum_squares / n) - (mean * mean);

        variance.sqrt() * (252.0_f64).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    pub count: usize,
    pub mean: f64,
    pub volatility: f64,
    pub min_return: f64,
    pub max_return: f64,
    pub latest_return: f64,
}

/// Optimized mathematical operations for financial analytics
pub mod math_ops {
    use super::*;

    /// Fast percentile calculation for risk metrics
    #[inline(always)]
    pub fn percentile(mut data: Vec<f64>, percentile: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Use unstable sort for speed (we don't need stable sorting)
        data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile / 100.0) * (data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            data[lower]
        } else {
            let weight = index - lower as f64;
            data[lower] * (1.0 - weight) + data[upper] * weight
        }
    }

    /// Value at Risk (VaR) calculation
    #[inline(always)]
    pub fn value_at_risk(returns: &[f64], confidence_level: f64) -> f64 {
        percentile(returns.to_vec(), (1.0 - confidence_level) * 100.0)
    }

    /// Expected Shortfall (CVaR) calculation
    #[inline(always)]
    pub fn expected_shortfall(returns: &[f64], confidence_level: f64) -> f64 {
        let var_threshold = value_at_risk(returns, confidence_level);

        let tail_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r <= var_threshold)
            .copied()
            .collect();

        if tail_returns.is_empty() {
            var_threshold
        } else {
            tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        }
    }

    /// Sharpe ratio calculation
    #[inline(always)]
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = mean_return - risk_free_rate / 252.0; // Daily risk-free rate
        let volatility =
            SIMDFinancialCalculator::volatility_optimized(returns) / (252.0_f64).sqrt(); // Daily volatility

        if volatility == 0.0 {
            0.0
        } else {
            excess_return / volatility
        }
    }

    /// Maximum drawdown calculation
    #[inline(always)]
    pub fn max_drawdown(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let mut peak = prices[0];
        let mut max_dd: f64 = 0.0;

        for &price in prices.iter().skip(1) {
            if price > peak {
                peak = price;
            } else {
                let drawdown = (peak - price) / peak;
                max_dd = max_dd.max(drawdown);
            }
        }

        max_dd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_daily_returns() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];
        let returns = SIMDFinancialCalculator::daily_returns_simd(&prices);

        assert_eq!(returns.len(), 4);
        assert!((returns[0] - 0.01).abs() < 1e-10); // (101-100)/100
    }

    #[test]
    fn test_volatility_consistency() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.008];
        let vol = SIMDFinancialCalculator::volatility_optimized(&returns);

        assert!(vol > 0.0);
        assert!(vol < 1.0); // Reasonable volatility range
    }

    #[test]
    fn test_correlation_bounds() {
        let returns1 = vec![0.01, 0.02, -0.01, 0.005];
        let returns2 = vec![0.015, 0.018, -0.012, 0.008];
        let corr = SIMDFinancialCalculator::correlation_simd(&returns1, &returns2);

        assert!(corr >= -1.0 && corr <= 1.0);
    }
}
