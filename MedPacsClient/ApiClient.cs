using System.Net;
using System.Net.Http.Json;
using System.Text.Json;

namespace MedPacsClient;

// ────────────────────────────────────────────────────────────────────────────────
//  ApiClient.cs  –  Typed HTTP client for the MedPACS-AI REST API
// ────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Strongly-typed HTTP client for the MedPACS-AI REST API running at
/// <c>http://localhost:8000</c>.  All methods implement transparent retry
/// with exponential back-off (up to <see cref="MaxRetries"/> attempts).
/// </summary>
public sealed class MedPacsApiClient : IDisposable
{
    // ── Constants ────────────────────────────────────────────────────────────

    /// <summary>Maximum number of retry attempts per request (excluding the first try).</summary>
    private const int MaxRetries = 3;

    /// <summary>Base delay for exponential back-off on transient failures.</summary>
    private static readonly TimeSpan BaseDelay = TimeSpan.FromMilliseconds(500);

    /// <summary>
    /// JSON serialiser options shared by all deserialisation calls.
    /// Property names on the wire are snake_case; we keep them as-is via
    /// <see cref="JsonPropertyName"/> attributes on the model records.
    /// </summary>
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition      = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    // ── Fields ───────────────────────────────────────────────────────────────

    private readonly HttpClient _http;
    private bool _disposed;

    // ── Constructor ──────────────────────────────────────────────────────────

    /// <summary>
    /// Initialises the client with an externally managed <see cref="HttpClient"/>.
    /// The caller is responsible for setting <see cref="HttpClient.BaseAddress"/>.
    /// </summary>
    /// <param name="httpClient">Configured <see cref="HttpClient"/> instance.</param>
    public MedPacsApiClient(HttpClient httpClient)
    {
        _http = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
    }

    // ── Public API methods ───────────────────────────────────────────────────

    /// <summary>
    /// Calls <c>GET /health</c> and returns the service health snapshot.
    /// </summary>
    /// <param name="ct">Optional cancellation token.</param>
    /// <returns>A <see cref="HealthResponse"/> describing current service state.</returns>
    /// <exception cref="ApiException">Thrown when the server returns a non-success status.</exception>
    public Task<HealthResponse> GetHealthAsync(CancellationToken ct = default)
        => SendWithRetryAsync<HealthResponse>(HttpMethod.Get, "health", body: null, ct);

    /// <summary>
    /// Calls <c>GET /series</c> and returns a list of all available DICOM series.
    /// </summary>
    /// <param name="ct">Optional cancellation token.</param>
    /// <returns>Ordered list of <see cref="SeriesSummary"/> records.</returns>
    public Task<List<SeriesSummary>> GetSeriesAsync(CancellationToken ct = default)
        => SendWithRetryAsync<List<SeriesSummary>>(HttpMethod.Get, "series", body: null, ct);

    /// <summary>
    /// Calls <c>GET /series/{uid}</c> and returns the full detail record for a series.
    /// </summary>
    /// <param name="uid">The DICOM Series Instance UID to look up.</param>
    /// <param name="ct">Optional cancellation token.</param>
    /// <returns>A <see cref="SeriesDetail"/> with all metadata and file paths.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="uid"/> is null or whitespace.</exception>
    public Task<SeriesDetail> GetSeriesByUidAsync(string uid, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(uid))
            throw new ArgumentException("Series UID must not be empty.", nameof(uid));

        return SendWithRetryAsync<SeriesDetail>(
            HttpMethod.Get, $"series/{Uri.EscapeDataString(uid)}", body: null, ct);
    }

    /// <summary>
    /// Calls <c>POST /pipeline/run</c> to enqueue a processing job for the given series.
    /// </summary>
    /// <param name="request">Job parameters including series UID, model, and priority.</param>
    /// <param name="ct">Optional cancellation token.</param>
    /// <returns>Initial <see cref="PipelineStatus"/> (status will be <c>queued</c>).</returns>
    public Task<PipelineStatus> RunPipelineAsync(PipelineRunRequest request, CancellationToken ct = default)
        => SendWithRetryAsync<PipelineStatus>(HttpMethod.Post, "pipeline/run", body: request, ct);

    /// <summary>
    /// Calls <c>GET /pipeline/status/{jobId}</c> to poll the current state of a job.
    /// </summary>
    /// <param name="jobId">Job identifier returned by <see cref="RunPipelineAsync"/>.</param>
    /// <param name="ct">Optional cancellation token.</param>
    /// <returns>Latest <see cref="PipelineStatus"/> snapshot.</returns>
    public Task<PipelineStatus> GetPipelineStatusAsync(string jobId, CancellationToken ct = default)
        => SendWithRetryAsync<PipelineStatus>(
            HttpMethod.Get, $"pipeline/status/{Uri.EscapeDataString(jobId)}", body: null, ct);

    /// <summary>
    /// Calls <c>GET /stats</c> and returns aggregate catalogue statistics.
    /// </summary>
    /// <param name="ct">Optional cancellation token.</param>
    /// <returns>A <see cref="Stats"/> record with counts and breakdowns.</returns>
    public Task<Stats> GetStatsAsync(CancellationToken ct = default)
        => SendWithRetryAsync<Stats>(HttpMethod.Get, "stats", body: null, ct);

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Sends an HTTP request and deserialises the JSON response into <typeparamref name="T"/>.
    /// Retries up to <see cref="MaxRetries"/> times on transient failures using
    /// exponential back-off with jitter.
    /// </summary>
    /// <typeparam name="T">Expected response type.</typeparam>
    /// <param name="method">HTTP verb.</param>
    /// <param name="relativeUrl">Endpoint path relative to the base address.</param>
    /// <param name="body">Optional request body (serialised to JSON if not null).</param>
    /// <param name="ct">Cancellation token.</param>
    private async Task<T> SendWithRetryAsync<T>(
        HttpMethod  method,
        string      relativeUrl,
        object?     body,
        CancellationToken ct)
    {
        for (int attempt = 0; attempt <= MaxRetries; attempt++)
        {
            ct.ThrowIfCancellationRequested();

            try
            {
                using var request = BuildRequest(method, relativeUrl, body);
                using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct)
                                               .ConfigureAwait(false);

                await EnsureSuccessAsync(response, ct).ConfigureAwait(false);

                var result = await response.Content
                    .ReadFromJsonAsync<T>(JsonOptions, ct)
                    .ConfigureAwait(false);

                return result ?? throw new ApiException(0, $"Server returned null for {relativeUrl}.");
            }
            catch (ApiException)
            {
                // Non-transient: re-throw immediately without retrying.
                throw;
            }
            catch (Exception ex) when (IsTransient(ex) && attempt < MaxRetries)
            {
                var delay = ComputeDelay(attempt);
                await Task.Delay(delay, ct).ConfigureAwait(false);
            }
        }

        // Unreachable – loop always returns or throws.
        throw new InvalidOperationException("Retry loop exited unexpectedly.");
    }

    /// <summary>Builds an <see cref="HttpRequestMessage"/> with an optional JSON body.</summary>
    private static HttpRequestMessage BuildRequest(HttpMethod method, string relativeUrl, object? body)
    {
        var request = new HttpRequestMessage(method, relativeUrl);

        if (body is not null)
            request.Content = JsonContent.Create(body, options: JsonOptions);

        return request;
    }

    /// <summary>
    /// Reads the response body and throws a descriptive <see cref="ApiException"/>
    /// if the status code indicates failure.
    /// </summary>
    private static async Task EnsureSuccessAsync(HttpResponseMessage response, CancellationToken ct)
    {
        if (response.IsSuccessStatusCode)
            return;

        string body = string.Empty;
        try { body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false); }
        catch { /* best-effort */ }

        throw new ApiException((int)response.StatusCode,
            $"HTTP {(int)response.StatusCode} {response.ReasonPhrase}: {body}");
    }

    /// <summary>
    /// Returns <see langword="true"/> for transient failures that are safe to retry.
    /// </summary>
    private static bool IsTransient(Exception ex) => ex is
        HttpRequestException                                        or
        TaskCanceledException { InnerException: TimeoutException } or
        OperationCanceledException;

    /// <summary>
    /// Computes an exponential back-off delay with ±10 % random jitter to
    /// avoid thundering-herd when many clients retry simultaneously.
    /// </summary>
    /// <param name="attempt">Zero-based attempt index.</param>
    private static TimeSpan ComputeDelay(int attempt)
    {
        double exponential = BaseDelay.TotalMilliseconds * Math.Pow(2, attempt);
        double jitter      = exponential * 0.1 * (Random.Shared.NextDouble() * 2 - 1);
        return TimeSpan.FromMilliseconds(Math.Min(exponential + jitter, 30_000));
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    /// <summary>Disposes the underlying <see cref="HttpClient"/> if owned by this instance.</summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _http.Dispose();
            _disposed = true;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────────
//  ApiException
// ────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Represents an error response received from the MedPACS-AI REST API.
/// </summary>
public sealed class ApiException : Exception
{
    /// <summary>The HTTP status code returned by the server.</summary>
    public int StatusCode { get; }

    /// <summary>
    /// Initialises a new <see cref="ApiException"/> with an HTTP status code and detail message.
    /// </summary>
    public ApiException(int statusCode, string message) : base(message)
        => StatusCode = statusCode;
}
