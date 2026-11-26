## Load Testing and Performance Evaluation

### Testing Environment

Load testing was performed using **Locust** to simulate multiple users accessing the GeoVision ML API.

The API was deployed using Docker Compose, with Nginx acting as a reverse proxy and distributing requests across API containers.

The experiments used the following configuration:

- Load testing tool: Locust
- Concurrent users: 10
- User spawn rate: 2 users per second
- Test duration: 2 minutes
- API endpoints: `/health` and `/predict`
- Container configurations: 1, 2, and 3 API containers

### Performance Results

| Metric | 1 Container | 2 Containers | 3 Containers |
|---|---:|---:|---:|
| Total Requests | 572 | 576 | 574 |
| Failed Requests | 0 | 0 | 0 |
| Average Response Time | 59 ms | 41 ms | 84 ms |
| P95 Response Time | 170 ms | 110 ms | 230 ms |
| Throughput | 4.78 req/s | 4.82 req/s | 4.81 req/s |

### Performance Analysis

All three configurations completed their tests without failed requests.

The two-container configuration achieved the lowest average response time of 41 ms and the lowest P95 response time of 110 ms.

Increasing the number of API containers to three did not improve performance in this experiment. The average response time increased to 84 ms.

Throughput remained almost unchanged across all three configurations, at approximately 4.8 requests per second.

These results suggest that increasing the number of containers does not automatically improve performance. Available CPU and memory resources, model inference workloads, and container overhead can influence the results.

### Conclusion

The two-container configuration provided the best response-time results in the initial experiments. Further repeated testing is necessary to determine whether this improvement is consistent.

### Test Evidence

Performance graphs and Locust output files are stored in:

`results/load_testing/`

- `replica_comparison.csv`
- `average_response_time.png`
- `p95_response_time.png`
- `throughput.png`
