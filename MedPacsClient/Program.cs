using System.Globalization;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Spectre.Console;

namespace MedPacsClient;

// ────────────────────────────────────────────────────────────────────────────────
//  Program.cs  –  Entry point for the MedPACS-AI console client
// ────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Application entry point.  Wires up DI, creates the <see cref="MedPacsApp"/>
/// instance and runs the main interaction loop.
/// </summary>
internal static class Program
{
    /// <summary>API base address (override with env var <c>MEDPACS_URL</c>).</summary>
    private static readonly string BaseUrl =
        Environment.GetEnvironmentVariable("MEDPACS_URL") ?? "http://localhost:8000";

    /// <summary>Application entry point.</summary>
    public static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        // ── DI container ─────────────────────────────────────────────────────
        var services = new ServiceCollection();

        services.AddHttpClient<MedPacsApiClient>(client =>
        {
            client.BaseAddress = new Uri(BaseUrl.TrimEnd('/') + "/");
            client.Timeout     = TimeSpan.FromSeconds(30);
            client.DefaultRequestHeaders.Add("User-Agent", "MedPacsClient/1.0");
            client.DefaultRequestHeaders.Add("Accept",     "application/json");
        });

        await using var provider = services.BuildServiceProvider();
        var apiClient = provider.GetRequiredService<MedPacsApiClient>();

        // ── Run the app ──────────────────────────────────────────────────────
        var app = new MedPacsApp(apiClient);
        await app.RunAsync();

        return 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────────
//  MedPacsApp  –  Main application controller
// ────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Orchestrates the interactive console experience for the MedPACS-AI client.
/// All rendering is done via Spectre.Console to ensure a rich, colour terminal UI.
/// </summary>
internal sealed class MedPacsApp
{
    private readonly MedPacsApiClient _api;

    /// <summary>Initialises the application with the provided API client.</summary>
    /// <param name="api">Configured <see cref="MedPacsApiClient"/>.</param>
    public MedPacsApp(MedPacsApiClient api) => _api = api;

    // ── Entry point ──────────────────────────────────────────────────────────

    /// <summary>
    /// Runs the full application lifecycle: splash screen → health check → menu loop.
    /// </summary>
    public async Task RunAsync()
    {
        RenderSplash();

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        var healthy = await CheckHealthAsync(cts.Token);
        if (!healthy)
        {
            AnsiConsole.MarkupLine("\n[bold red]Cannot connect to MedPACS-AI API. Exiting.[/]");
            return;
        }

        await RunMenuLoopAsync(cts.Token);
    }

    // ── Splash screen ─────────────────────────────────────────────────────────

    /// <summary>Renders the ASCII art splash/banner on startup.</summary>
    private static void RenderSplash()
    {
        AnsiConsole.Clear();
        AnsiConsole.Write(new FigletText("MedPACS-AI")
            .Centered()
            .Color(Color.DeepSkyBlue1));

        AnsiConsole.Write(new Rule("[bold grey50]DICOM Intelligence Platform  •  Console Client v1.0[/]")
            .RuleStyle(Style.Parse("grey50")));

        AnsiConsole.WriteLine();
    }

    // ── Health check ──────────────────────────────────────────────────────────

    /// <summary>
    /// Calls <c>GET /health</c>, renders a status panel, and returns whether the
    /// service is reachable and healthy.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    private async Task<bool> CheckHealthAsync(CancellationToken ct)
    {
        HealthResponse? health = null;

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots12)
            .SpinnerStyle(Style.Parse("deepskyblue1"))
            .StartAsync("[deepskyblue1]Connecting to MedPACS-AI API…[/]", async ctx =>
            {
                ctx.Status("[deepskyblue1]Calling GET /health …[/]");
                try
                {
                    health = await _api.GetHealthAsync(ct);
                }
                catch (Exception ex)
                {
                    AnsiConsole.MarkupLine($"[red]Error: {Markup.Escape(ex.Message)}[/]");
                }
            });

        if (health is null) return false;

        // ── Render status panel ──────────────────────────────────────────────
        var statusColour = health.Status.Equals("healthy", StringComparison.OrdinalIgnoreCase)
            ? "green" : "yellow";
        var gpuLabel = health.GpuAvailable
            ? "[green]✔ Available[/]" : "[red]✘ Not detected[/]";

        var panel = new Panel(
            new Rows(
                new Markup($"[bold]Status  :[/]  [{statusColour}]{Markup.Escape(health.Status.ToUpperInvariant())}[/]"),
                new Markup($"[bold]Version :[/]  [white]{Markup.Escape(health.Version)}[/]"),
                new Markup($"[bold]Uptime  :[/]  [white]{FormatUptime(health.Uptime)}[/]"),
                new Markup($"[bold]DICOM   :[/]  [white]{health.DicomLoaded} series loaded[/]"),
                new Markup($"[bold]GPU     :[/]  {gpuLabel}")
            ))
        {
            Header  = new PanelHeader("[bold deepskyblue1]  Service Health  [/]"),
            Border  = BoxBorder.Rounded,
            Padding = new Padding(1, 0)
        };

        AnsiConsole.Write(panel);
        AnsiConsole.WriteLine();

        return health.Status.Equals("healthy", StringComparison.OrdinalIgnoreCase);
    }

    // ── Main menu loop ────────────────────────────────────────────────────────

    /// <summary>Displays the main menu and dispatches the user's selection in a loop.</summary>
    /// <param name="ct">Cancellation token.</param>
    private async Task RunMenuLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            AnsiConsole.Write(new Rule("[bold deepskyblue1]Main Menu[/]").RuleStyle(Style.Parse("grey50")));
            AnsiConsole.WriteLine();

            var choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[bold white]What would you like to do?[/]")
                    .PageSize(10)
                    .HighlightStyle(Style.Parse("bold deepskyblue1"))
                    .AddChoices(
                        "[1] List all series",
                        "[2] View series detail",
                        "[3] Run AI pipeline",
                        "[4] Show statistics",
                        "[5] Export to CSV",
                        "[0] Exit"));

            AnsiConsole.WriteLine();

            try
            {
                await (choice[1] switch
                {
                    '1' => ListSeriesAsync(ct),
                    '2' => ViewSeriesDetailAsync(ct),
                    '3' => RunPipelineAsync(ct),
                    '4' => ShowStatsAsync(ct),
                    '5' => ExportToCsvAsync(ct),
                    '0' => ExitApp(ct),
                    _   => Task.CompletedTask
                });
            }
            catch (OperationCanceledException) { break; }
            catch (ApiException ex)
            {
                RenderError($"API Error ({ex.StatusCode})", ex.Message);
            }
            catch (HttpRequestException ex)
            {
                RenderError("Connection Error", ex.Message);
            }
            catch (Exception ex)
            {
                RenderError("Unexpected Error", ex.ToString());
            }

            AnsiConsole.WriteLine();
        }

        AnsiConsole.MarkupLine("[grey50]Goodbye.[/]");
    }

    // ── Option 1 – List all series ─────────────────────────────────────────────

    /// <summary>Fetches all series and renders them in a rich Spectre.Console table.</summary>
    /// <param name="ct">Cancellation token.</param>
    private async Task ListSeriesAsync(CancellationToken ct)
    {
        List<SeriesSummary>? series = null;

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots12)
            .SpinnerStyle(Style.Parse("deepskyblue1"))
            .StartAsync("[deepskyblue1]Fetching series list…[/]", async _ =>
            {
                series = await _api.GetSeriesAsync(ct);
            });

        if (series is null || series.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No series found in the catalogue.[/]");
            return;
        }

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderStyle(Style.Parse("grey50"))
            .Title($"[bold deepskyblue1]DICOM Series Catalogue[/]  [grey50]({series.Count} series)[/]")
            .AddColumn(new TableColumn("[bold]#[/]").RightAligned())
            .AddColumn(new TableColumn("[bold]Patient[/]"))
            .AddColumn(new TableColumn("[bold]Modality[/]").Centered())
            .AddColumn(new TableColumn("[bold]Study Date[/]").Centered())
            .AddColumn(new TableColumn("[bold]Description[/]"))
            .AddColumn(new TableColumn("[bold]Slices[/]").RightAligned())
            .AddColumn(new TableColumn("[bold]Spacing (mm)[/]").Centered())
            .AddColumn(new TableColumn("[bold]Status[/]").Centered())
            .AddColumn(new TableColumn("[bold]Size[/]").RightAligned());

        for (int i = 0; i < series.Count; i++)
        {
            var s         = series[i];
            var colour    = ModelExtensions.StatusColour(s.PipelineStatus);
            var spacing   = s.SliceThickness.HasValue
                ? $"{s.SliceThickness:F2} / {s.PixelSpacing ?? "?"}"
                : "N/A";

            table.AddRow(
                $"[grey50]{i + 1}[/]",
                Markup.Escape($"{s.PatientName} ({s.PatientId})"),
                $"[bold cyan]{Markup.Escape(s.Modality)}[/]",
                ModelExtensions.FormatDicomDate(s.StudyDate),
                $"[grey84]{Markup.Escape(Truncate(s.SeriesDescription, 32))}[/]",
                $"[white]{s.NumberOfSlices}[/]",
                $"[grey84]{Markup.Escape(spacing)}[/]",
                $"[{colour}]{Markup.Escape(s.PipelineStatus)}[/]",
                ModelExtensions.FormatBytes(s.FileSizeBytes));
        }

        AnsiConsole.Write(table);
    }

    // ── Option 2 – View series detail ─────────────────────────────────────────

    /// <summary>Prompts for a Series UID and renders the full detail record.</summary>
    /// <param name="ct">Cancellation token.</param>
    private async Task ViewSeriesDetailAsync(CancellationToken ct)
    {
        var uid = AnsiConsole.Ask<string>("[deepskyblue1]Enter Series Instance UID:[/]").Trim();
        if (string.IsNullOrEmpty(uid)) return;

        SeriesDetail? detail = null;

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots12)
            .SpinnerStyle(Style.Parse("deepskyblue1"))
            .StartAsync("[deepskyblue1]Loading series detail…[/]", async _ =>
            {
                detail = await _api.GetSeriesByUidAsync(uid, ct);
            });

        if (detail is null) return;

        var colour = ModelExtensions.StatusColour(detail.PipelineStatus);

        // ── Header panel ─────────────────────────────────────────────────────
        var header = new Panel(
            new Rows(
                new Markup($"[bold]Series UID  :[/]  [grey84]{Markup.Escape(detail.SeriesInstanceUid)}[/]"),
                new Markup($"[bold]Patient     :[/]  [white]{Markup.Escape(detail.PatientName)} " +
                           $"[grey50]({Markup.Escape(detail.PatientId)})[/]"),
                new Markup($"[bold]Age / Sex   :[/]  [white]{detail.PatientAge ?? "N/A"} / {detail.PatientSex ?? "N/A"}[/]")
            ))
        {
            Header  = new PanelHeader("[bold deepskyblue1]  Patient  [/]"),
            Border  = BoxBorder.Rounded,
            Padding = new Padding(1, 0)
        };

        AnsiConsole.Write(header);

        // ── Metadata grid ────────────────────────────────────────────────────
        var grid = new Grid()
            .AddColumn()
            .AddColumn();

        grid.AddRow("[bold]Modality[/]",        $"[cyan]{Markup.Escape(detail.Modality)}[/]");
        grid.AddRow("[bold]Study Date[/]",      ModelExtensions.FormatDicomDate(detail.StudyDate));
        grid.AddRow("[bold]Study Time[/]",      detail.StudyTime ?? "N/A");
        grid.AddRow("[bold]Study Desc.[/]",     Markup.Escape(detail.StudyDescription ?? "N/A"));
        grid.AddRow("[bold]Series Desc.[/]",    Markup.Escape(detail.SeriesDescription));
        grid.AddRow("[bold]Institution[/]",     Markup.Escape(detail.InstitutionName ?? "N/A"));
        grid.AddRow("[bold]Manufacturer[/]",    Markup.Escape($"{detail.Manufacturer ?? "N/A"} / {detail.ManufacturerModelName ?? "N/A"}"));
        grid.AddRow("[bold]kVp[/]",             detail.KvP.HasValue ? $"{detail.KvP:F0} kV" : "N/A");
        grid.AddRow("[bold]Dimensions[/]",      $"{detail.Columns} × {detail.Rows} × {detail.NumberOfSlices} px");
        grid.AddRow("[bold]Slice Thickness[/]", detail.SliceThickness.HasValue ? $"{detail.SliceThickness:F2} mm" : "N/A");
        grid.AddRow("[bold]Pixel Spacing[/]",   Markup.Escape(detail.PixelSpacing ?? "N/A"));
        grid.AddRow("[bold]Pipeline Status[/]", $"[{colour}]{Markup.Escape(detail.PipelineStatus)}[/]");
        grid.AddRow("[bold]File Size[/]",       ModelExtensions.FormatBytes(detail.FileSizeBytes));
        grid.AddRow("[bold]Files[/]",           $"{detail.FilePaths.Count} DICOM files");

        AnsiConsole.Write(new Panel(grid)
        {
            Header  = new PanelHeader("[bold deepskyblue1]  Acquisition Metadata  [/]"),
            Border  = BoxBorder.Rounded,
            Padding = new Padding(1, 0)
        });

        // ── Extra tags ───────────────────────────────────────────────────────
        if (detail.Tags is { Count: > 0 } tags)
        {
            var tagTable = new Table()
                .Border(TableBorder.Simple)
                .AddColumn("[bold]DICOM Tag[/]")
                .AddColumn("[bold]Value[/]");

            foreach (var (key, val) in tags)
                tagTable.AddRow(Markup.Escape(key), Markup.Escape(Truncate(val, 60)));

            AnsiConsole.Write(new Panel(tagTable)
            {
                Header  = new PanelHeader("[bold deepskyblue1]  Extended Tags  [/]"),
                Border  = BoxBorder.Rounded,
                Padding = new Padding(1, 0)
            });
        }
    }

    // ── Option 3 – Run AI pipeline ────────────────────────────────────────────

    /// <summary>
    /// Prompts for a Series UID, enqueues a pipeline job, then polls and renders
    /// a live progress bar until the job completes or fails.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    private async Task RunPipelineAsync(CancellationToken ct)
    {
        var uid = AnsiConsole.Ask<string>("[deepskyblue1]Enter Series Instance UID to process:[/]").Trim();
        if (string.IsNullOrEmpty(uid)) return;

        var model = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("[bold white]Select AI model:[/]")
                .HighlightStyle(Style.Parse("bold deepskyblue1"))
                .AddChoices("default", "high_res", "fast"));

        var priority = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("[bold white]Select priority:[/]")
                .HighlightStyle(Style.Parse("bold deepskyblue1"))
                .AddChoices("normal", "high", "low"));

        // Enqueue job
        PipelineStatus? status = null;
        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots12)
            .SpinnerStyle(Style.Parse("deepskyblue1"))
            .StartAsync("[deepskyblue1]Submitting pipeline job…[/]", async _ =>
            {
                status = await _api.RunPipelineAsync(
                    new PipelineRunRequest(uid, model, priority), ct);
            });

        if (status is null) return;

        AnsiConsole.MarkupLine($"[green]✔ Job enqueued:[/]  [bold white]{status.JobId}[/]");
        AnsiConsole.WriteLine();

        // ── Live polling with progress bar ───────────────────────────────────
        await AnsiConsole.Progress()
            .AutoClear(false)
            .Columns(
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new ElapsedTimeColumn(),
                new SpinnerColumn(Spinner.Known.Dots12))
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[deepskyblue1]Pipeline processing[/]", maxValue: 100);

                while (!ct.IsCancellationRequested)
                {
                    status = await _api.GetPipelineStatusAsync(status!.JobId, ct);

                    task.Value       = status.ProgressPercent;
                    task.Description = $"[deepskyblue1]{Markup.Escape(status.Stage)}[/]";

                    if (status.Status is "completed" or "failed")
                    {
                        task.Value = 100;
                        break;
                    }

                    await Task.Delay(2_000, ct);
                }
            });

        AnsiConsole.WriteLine();

        if (status!.Status == "completed")
        {
            AnsiConsole.MarkupLine($"[bold green]✔ Pipeline completed successfully![/]");
            if (status.OutputPath is not null)
                AnsiConsole.MarkupLine($"[bold]Output:[/]  [grey84]{Markup.Escape(status.OutputPath)}[/]");
        }
        else
        {
            AnsiConsole.MarkupLine($"[bold red]✘ Pipeline failed![/]");
            if (status.ErrorMessage is not null)
                AnsiConsole.MarkupLine($"[red]{Markup.Escape(status.ErrorMessage)}[/]");
        }
    }

    // ── Option 4 – Show statistics ────────────────────────────────────────────

    /// <summary>Fetches catalogue statistics and renders a summary dashboard.</summary>
    /// <param name="ct">Cancellation token.</param>
    private async Task ShowStatsAsync(CancellationToken ct)
    {
        Stats? stats = null;

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots12)
            .SpinnerStyle(Style.Parse("deepskyblue1"))
            .StartAsync("[deepskyblue1]Fetching statistics…[/]", async _ =>
            {
                stats = await _api.GetStatsAsync(ct);
            });

        if (stats is null) return;

        // ── Summary panel ─────────────────────────────────────────────────────
        var summary = new Panel(
            new Rows(
                new Markup($"[bold]Total Series    :[/]  [white]{stats.TotalSeries:N0}[/]"),
                new Markup($"[bold]Total Patients  :[/]  [white]{stats.TotalPatients:N0}[/]"),
                new Markup($"[bold]Total Slices    :[/]  [white]{stats.TotalSlices:N0}[/]"),
                new Markup($"[bold]Total Size      :[/]  [white]{ModelExtensions.FormatBytes(stats.TotalFileSizeBytes)}[/]"),
                new Markup($"[bold]Avg Slices/Ser. :[/]  [white]{stats.AverageSlicesPerSeries:F1}[/]"),
                new Markup($"[bold]Date Range      :[/]  " +
                           $"[white]{ModelExtensions.FormatDicomDate(stats.OldestStudyDate)} " +
                           $"→ {ModelExtensions.FormatDicomDate(stats.NewestStudyDate)}[/]")))
        {
            Header  = new PanelHeader("[bold deepskyblue1]  Catalogue Summary  [/]"),
            Border  = BoxBorder.Rounded,
            Padding = new Padding(1, 0)
        };

        AnsiConsole.Write(summary);

        // ── Modality breakdown table ───────────────────────────────────────────
        if (stats.ModalityBreakdown.Count > 0)
        {
            var modTable = new Table()
                .Border(TableBorder.Rounded)
                .BorderStyle(Style.Parse("grey50"))
                .Title("[bold deepskyblue1]Modality Breakdown[/]")
                .AddColumn("[bold]Modality[/]")
                .AddColumn(new TableColumn("[bold]Count[/]").RightAligned())
                .AddColumn(new TableColumn("[bold]Share[/]").RightAligned());

            foreach (var (modality, count) in stats.ModalityBreakdown.OrderByDescending(x => x.Value))
            {
                double pct = stats.TotalSeries > 0 ? 100.0 * count / stats.TotalSeries : 0;
                modTable.AddRow(
                    $"[cyan]{Markup.Escape(modality)}[/]",
                    $"[white]{count}[/]",
                    $"[grey84]{pct:F1} %[/]");
            }

            AnsiConsole.Write(modTable);
        }

        // ── Pipeline status breakdown ─────────────────────────────────────────
        if (stats.StatusBreakdown.Count > 0)
        {
            var statusTable = new Table()
                .Border(TableBorder.Rounded)
                .BorderStyle(Style.Parse("grey50"))
                .Title("[bold deepskyblue1]Pipeline Status Breakdown[/]")
                .AddColumn("[bold]Status[/]")
                .AddColumn(new TableColumn("[bold]Count[/]").RightAligned());

            foreach (var (st, count) in stats.StatusBreakdown.OrderByDescending(x => x.Value))
            {
                var c = ModelExtensions.StatusColour(st);
                statusTable.AddRow($"[{c}]{Markup.Escape(st)}[/]", $"[white]{count}[/]");
            }

            AnsiConsole.Write(statusTable);
        }
    }

    // ── Option 5 – Export to CSV ──────────────────────────────────────────────

    /// <summary>
    /// Fetches the complete series list and writes it to <c>medpacs_export.csv</c>
    /// in the current working directory.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    private async Task ExportToCsvAsync(CancellationToken ct)
    {
        List<SeriesSummary>? series = null;

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots12)
            .SpinnerStyle(Style.Parse("deepskyblue1"))
            .StartAsync("[deepskyblue1]Fetching series for export…[/]", async _ =>
            {
                series = await _api.GetSeriesAsync(ct);
            });

        if (series is null || series.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No series to export.[/]");
            return;
        }

        const string path = "medpacs_export.csv";

        await AnsiConsole.Progress()
            .AutoClear(false)
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn())
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[deepskyblue1]Writing CSV[/]", maxValue: series.Count);

                await using var writer = new StreamWriter(path, append: false, Encoding.UTF8);

                // ── Header row ────────────────────────────────────────────────
                await writer.WriteLineAsync(
                    "SeriesInstanceUID,PatientID,PatientName,Modality,StudyDate," +
                    "SeriesDescription,NumberOfSlices,SliceThickness_mm,PixelSpacing," +
                    "PipelineStatus,FileSizeBytes");

                // ── Data rows ──────────────────────────────────────────────────
                foreach (var s in series)
                {
                    await writer.WriteLineAsync(string.Join(",",
                        CsvEscape(s.SeriesInstanceUid),
                        CsvEscape(s.PatientId),
                        CsvEscape(s.PatientName),
                        CsvEscape(s.Modality),
                        CsvEscape(s.StudyDate),
                        CsvEscape(s.SeriesDescription),
                        s.NumberOfSlices.ToString(CultureInfo.InvariantCulture),
                        s.SliceThickness?.ToString("F4", CultureInfo.InvariantCulture) ?? "",
                        CsvEscape(s.PixelSpacing ?? ""),
                        CsvEscape(s.PipelineStatus),
                        s.FileSizeBytes.ToString(CultureInfo.InvariantCulture)));

                    task.Increment(1);
                }
            });

        var fullPath = Path.GetFullPath(path);
        AnsiConsole.MarkupLine(
            $"[green]✔ Exported {series.Count} series to:[/]  [bold white]{Markup.Escape(fullPath)}[/]");
    }

    // ── Option 0 – Exit ───────────────────────────────────────────────────────

    /// <summary>Displays a farewell message and cancels the main loop.</summary>
    private static Task ExitApp(CancellationToken ct)
    {
        AnsiConsole.MarkupLine("[bold deepskyblue1]Shutting down MedPACS-AI client…[/]");
        throw new OperationCanceledException(ct);
    }

    // ── Utility helpers ───────────────────────────────────────────────────────

    /// <summary>Renders a styled error panel for unexpected exceptions.</summary>
    private static void RenderError(string title, string message)
    {
        AnsiConsole.Write(new Panel(
            new Markup($"[red]{Markup.Escape(message)}[/]"))
        {
            Header  = new PanelHeader($"[bold red]  ✘ {Markup.Escape(title)}  [/]"),
            Border  = BoxBorder.Rounded,
            Padding = new Padding(1, 0)
        });
    }

    /// <summary>Truncates a string to <paramref name="maxLen"/> characters, appending "…" if needed.</summary>
    private static string Truncate(string? s, int maxLen)
    {
        if (s is null || s.Length <= maxLen) return s ?? string.Empty;
        return s[..(maxLen - 1)] + "…";
    }

    /// <summary>Wraps a CSV field in double quotes, escaping any embedded double quotes.</summary>
    private static string CsvEscape(string value)
        => $"\"{value.Replace("\"", "\"\"")}\"";

    /// <summary>Converts a seconds value to a human-readable uptime string.</summary>
    private static string FormatUptime(double seconds)
    {
        var ts = TimeSpan.FromSeconds(seconds);
        return ts.TotalDays >= 1
            ? $"{(int)ts.TotalDays}d {ts.Hours:D2}h {ts.Minutes:D2}m"
            : ts.TotalHours >= 1
                ? $"{ts.Hours}h {ts.Minutes:D2}m {ts.Seconds:D2}s"
                : $"{ts.Minutes}m {ts.Seconds:D2}s";
    }
}
