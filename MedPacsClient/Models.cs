using System.Text.Json.Serialization;

namespace MedPacsClient;

// ────────────────────────────────────────────────────────────────────────────────
//  Models.cs  –  C# record types matching the MedPACS-AI FastAPI Pydantic models
// ────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Lightweight health-check response returned by <c>GET /health</c>.
/// </summary>
/// <param name="Status">Overall service status string (e.g., "healthy").</param>
/// <param name="Version">Application version string (e.g., "1.0.0").</param>
/// <param name="Uptime">Server uptime in seconds.</param>
/// <param name="DicomLoaded">Number of DICOM series currently loaded in memory.</param>
/// <param name="GpuAvailable">Whether a CUDA-capable GPU was detected at startup.</param>
public record HealthResponse(
    [property: JsonPropertyName("status")]        string  Status,
    [property: JsonPropertyName("version")]       string  Version,
    [property: JsonPropertyName("uptime")]        double  Uptime,
    [property: JsonPropertyName("dicom_loaded")]  int     DicomLoaded,
    [property: JsonPropertyName("gpu_available")] bool    GpuAvailable
);

/// <summary>
/// Summary row for a single DICOM series, returned by <c>GET /series</c>.
/// </summary>
/// <param name="SeriesInstanceUid">Globally-unique DICOM Series Instance UID.</param>
/// <param name="PatientId">Patient identifier from the DICOM header.</param>
/// <param name="PatientName">Patient name from the DICOM header.</param>
/// <param name="Modality">Imaging modality (CT, MR, PT, …).</param>
/// <param name="StudyDate">Study acquisition date in YYYYMMDD format.</param>
/// <param name="SeriesDescription">Human-readable series description tag.</param>
/// <param name="NumberOfSlices">Total slice count in the series.</param>
/// <param name="SliceThickness">Inter-slice spacing in millimetres.</param>
/// <param name="PixelSpacing">In-plane pixel spacing as a comma-separated pair (e.g. "0.98,0.98").</param>
/// <param name="PipelineStatus">Last known pipeline processing status for this series.</param>
/// <param name="FileSizeBytes">Total on-disk size of all DICOM files in the series.</param>
public record SeriesSummary(
    [property: JsonPropertyName("series_instance_uid")] string  SeriesInstanceUid,
    [property: JsonPropertyName("patient_id")]          string  PatientId,
    [property: JsonPropertyName("patient_name")]        string  PatientName,
    [property: JsonPropertyName("modality")]            string  Modality,
    [property: JsonPropertyName("study_date")]          string  StudyDate,
    [property: JsonPropertyName("series_description")]  string  SeriesDescription,
    [property: JsonPropertyName("number_of_slices")]    int     NumberOfSlices,
    [property: JsonPropertyName("slice_thickness")]     double? SliceThickness,
    [property: JsonPropertyName("pixel_spacing")]       string? PixelSpacing,
    [property: JsonPropertyName("pipeline_status")]     string  PipelineStatus,
    [property: JsonPropertyName("file_size_bytes")]     long    FileSizeBytes
);

/// <summary>
/// Detailed view of a single DICOM series, returned by <c>GET /series/{uid}</c>.
/// Extends <see cref="SeriesSummary"/> with acquisition and reconstruction metadata.
/// </summary>
/// <param name="SeriesInstanceUid">Globally-unique DICOM Series Instance UID.</param>
/// <param name="PatientId">Patient identifier from the DICOM header.</param>
/// <param name="PatientName">Patient name from the DICOM header.</param>
/// <param name="PatientAge">Patient age string from the DICOM header (e.g. "045Y").</param>
/// <param name="PatientSex">Patient sex from the DICOM header (M / F / O).</param>
/// <param name="Modality">Imaging modality (CT, MR, PT, …).</param>
/// <param name="StudyDate">Study acquisition date in YYYYMMDD format.</param>
/// <param name="StudyTime">Study acquisition time in HHMMSS format.</param>
/// <param name="StudyDescription">Human-readable study description tag.</param>
/// <param name="SeriesDescription">Human-readable series description tag.</param>
/// <param name="InstitutionName">Name of the institution where the study was performed.</param>
/// <param name="Manufacturer">Equipment manufacturer.</param>
/// <param name="ManufacturerModelName">Equipment model name.</param>
/// <param name="KvP">Peak kilovoltage (CT only).</param>
/// <param name="NumberOfSlices">Total slice count in the series.</param>
/// <param name="SliceThickness">Inter-slice spacing in millimetres.</param>
/// <param name="PixelSpacing">In-plane pixel spacing as a comma-separated pair.</param>
/// <param name="Rows">Image height in pixels.</param>
/// <param name="Columns">Image width in pixels.</param>
/// <param name="PipelineStatus">Last known pipeline processing status for this series.</param>
/// <param name="FileSizeBytes">Total on-disk size of all DICOM files in the series.</param>
/// <param name="FilePaths">Sorted list of absolute paths to the individual DICOM files.</param>
/// <param name="Tags">Arbitrary additional DICOM tags as a key-value map.</param>
public record SeriesDetail(
    [property: JsonPropertyName("series_instance_uid")]    string               SeriesInstanceUid,
    [property: JsonPropertyName("patient_id")]             string               PatientId,
    [property: JsonPropertyName("patient_name")]           string               PatientName,
    [property: JsonPropertyName("patient_age")]            string?              PatientAge,
    [property: JsonPropertyName("patient_sex")]            string?              PatientSex,
    [property: JsonPropertyName("modality")]               string               Modality,
    [property: JsonPropertyName("study_date")]             string               StudyDate,
    [property: JsonPropertyName("study_time")]             string?              StudyTime,
    [property: JsonPropertyName("study_description")]      string?              StudyDescription,
    [property: JsonPropertyName("series_description")]     string               SeriesDescription,
    [property: JsonPropertyName("institution_name")]       string?              InstitutionName,
    [property: JsonPropertyName("manufacturer")]           string?              Manufacturer,
    [property: JsonPropertyName("manufacturer_model")]     string?              ManufacturerModelName,
    [property: JsonPropertyName("kvp")]                    double?              KvP,
    [property: JsonPropertyName("number_of_slices")]       int                  NumberOfSlices,
    [property: JsonPropertyName("slice_thickness")]        double?              SliceThickness,
    [property: JsonPropertyName("pixel_spacing")]          string?              PixelSpacing,
    [property: JsonPropertyName("rows")]                   int                  Rows,
    [property: JsonPropertyName("columns")]                int                  Columns,
    [property: JsonPropertyName("pipeline_status")]        string               PipelineStatus,
    [property: JsonPropertyName("file_size_bytes")]        long                 FileSizeBytes,
    [property: JsonPropertyName("file_paths")]             List<string>         FilePaths,
    [property: JsonPropertyName("tags")]                   Dictionary<string,string>? Tags
);

/// <summary>
/// Snapshot of a running or completed pipeline job, returned by
/// <c>POST /pipeline/run</c> and polled via <c>GET /pipeline/status/{job_id}</c>.
/// </summary>
/// <param name="JobId">Unique job identifier assigned by the server.</param>
/// <param name="SeriesUid">The series UID that this job is processing.</param>
/// <param name="Status">Current status: <c>queued</c>, <c>running</c>, <c>completed</c>, or <c>failed</c>.</param>
/// <param name="Stage">Human-readable description of the current processing stage.</param>
/// <param name="ProgressPercent">Completion percentage in the range [0, 100].</param>
/// <param name="StartedAt">ISO-8601 timestamp when the job started (null if still queued).</param>
/// <param name="FinishedAt">ISO-8601 timestamp when the job finished (null if still running).</param>
/// <param name="ErrorMessage">Error details if the job failed, otherwise null.</param>
/// <param name="OutputPath">Path to the pipeline output artefacts (available once completed).</param>
public record PipelineStatus(
    [property: JsonPropertyName("job_id")]           string  JobId,
    [property: JsonPropertyName("series_uid")]       string  SeriesUid,
    [property: JsonPropertyName("status")]           string  Status,
    [property: JsonPropertyName("stage")]            string  Stage,
    [property: JsonPropertyName("progress_percent")] double  ProgressPercent,
    [property: JsonPropertyName("started_at")]       string? StartedAt,
    [property: JsonPropertyName("finished_at")]      string? FinishedAt,
    [property: JsonPropertyName("error_message")]    string? ErrorMessage,
    [property: JsonPropertyName("output_path")]      string? OutputPath
);

/// <summary>
/// Request body for <c>POST /pipeline/run</c>.
/// </summary>
/// <param name="SeriesUid">The series to process.</param>
/// <param name="Model">The AI model variant to use (e.g., "default", "high_res").</param>
/// <param name="Priority">Job priority: <c>low</c>, <c>normal</c>, or <c>high</c>.</param>
public record PipelineRunRequest(
    [property: JsonPropertyName("series_uid")] string SeriesUid,
    [property: JsonPropertyName("model")]      string Model    = "default",
    [property: JsonPropertyName("priority")]   string Priority = "normal"
);

/// <summary>
/// Aggregate statistics for the entire DICOM catalogue, returned by <c>GET /stats</c>.
/// </summary>
/// <param name="TotalSeries">Total number of series in the catalogue.</param>
/// <param name="TotalPatients">Number of unique patients.</param>
/// <param name="TotalSlices">Cumulative slice count across all series.</param>
/// <param name="TotalFileSizeBytes">Cumulative file size in bytes across all series.</param>
/// <param name="ModalityBreakdown">Counts per modality (e.g. {"CT": 42, "MR": 17}).</param>
/// <param name="StatusBreakdown">Counts per pipeline status.</param>
/// <param name="OldestStudyDate">Date of the oldest study in the catalogue (YYYYMMDD).</param>
/// <param name="NewestStudyDate">Date of the most recent study in the catalogue (YYYYMMDD).</param>
/// <param name="AverageSlicesPerSeries">Mean slice count per series.</param>
public record Stats(
    [property: JsonPropertyName("total_series")]             int                      TotalSeries,
    [property: JsonPropertyName("total_patients")]           int                      TotalPatients,
    [property: JsonPropertyName("total_slices")]             long                     TotalSlices,
    [property: JsonPropertyName("total_file_size_bytes")]    long                     TotalFileSizeBytes,
    [property: JsonPropertyName("modality_breakdown")]       Dictionary<string, int>  ModalityBreakdown,
    [property: JsonPropertyName("status_breakdown")]         Dictionary<string, int>  StatusBreakdown,
    [property: JsonPropertyName("oldest_study_date")]        string?                  OldestStudyDate,
    [property: JsonPropertyName("newest_study_date")]        string?                  NewestStudyDate,
    [property: JsonPropertyName("average_slices_per_series")]double                   AverageSlicesPerSeries
);

/// <summary>Utility helpers for the model layer.</summary>
internal static class ModelExtensions
{
    /// <summary>Formats a byte count as a human-readable string (KB / MB / GB).</summary>
    public static string FormatBytes(long bytes)
    {
        const long kb = 1024;
        const long mb = kb * 1024;
        const long gb = mb * 1024;

        return bytes switch
        {
            >= gb => $"{bytes / (double)gb:F2} GB",
            >= mb => $"{bytes / (double)mb:F2} MB",
            >= kb => $"{bytes / (double)kb:F2} KB",
            _     => $"{bytes} B"
        };
    }

    /// <summary>Converts a YYYYMMDD DICOM date string to a display-friendly format.</summary>
    public static string FormatDicomDate(string? raw)
    {
        if (raw is null || raw.Length != 8) return raw ?? "N/A";
        return $"{raw[0..4]}-{raw[4..6]}-{raw[6..8]}";
    }

    /// <summary>Returns a Spectre.Console markup colour tag for a pipeline status value.</summary>
    public static string StatusColour(string status) => status.ToLowerInvariant() switch
    {
        "completed" => "green",
        "running"   => "yellow",
        "queued"    => "blue",
        "failed"    => "red",
        _           => "grey"
    };
}
