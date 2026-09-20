//
//  ContentView.swift
//  VaaniDemo
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Vaani
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var model = RecognizerModel()
    @State private var importing = false

    var body: some View {
        NavigationStack {
            List {
                modelSection
                inputSection
                liveSection
                benchmarkSection
                if !model.outcomes.isEmpty { resultsSection }
                settingsSection
            }
            .navigationTitle("Vaani")
            .task { await model.loadIfCached() }
            .fileImporter(isPresented: $importing,
                          allowedContentTypes: [.audio],
                          allowsMultipleSelection: false) { result in
                if case .success(let urls) = result, let url = urls.first {
                    Task { await model.transcribe(fileAt: url) }
                }
            }
        }
    }

    private var modelSection: some View {
        Section("Model") {
            switch model.state {
            case .needsModel:
                Button { Task { await model.load() } } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }

            case .downloading(let file, let fraction, let index, let count):
                VStack(alignment: .leading, spacing: 6) {
                    Text("Downloading \(index + 1) of \(count) files").font(.subheadline)
                    Text(file).font(.caption).foregroundStyle(.secondary)
                    ProgressView(value: fraction)
                    Text("\(Int(fraction * 100))%").font(.caption2).foregroundStyle(.secondary)
                }

            case .loading:
                Label("Loading weights", systemImage: "cpu").foregroundStyle(.secondary)

            case .ready:
                Label("\(model.language.name) · \(model.decoder.rawValue.uppercased())",
                      systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)

            case .working:
                HStack { ProgressView(); Text("Transcribing").foregroundStyle(.secondary) }

            case .failed(let message):
                VStack(alignment: .leading, spacing: 4) {
                    Label("Failed", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(message).font(.caption).foregroundStyle(.secondary)
                    Button("Try again") { Task { await model.load() } }.font(.caption)
                }
            }

            if model.cachedBytes > 0 {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Cached on device")
                        Spacer()
                        Text(ByteCountFormatter.string(fromByteCount: model.cachedBytes,
                                                       countStyle: .file))
                    }
                    Text(model.origin)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private var inputSection: some View {
        Section("Transcribe") {
            ForEach(model.samples) { sample in
                Button { Task { await model.transcribe(sample) } } label: {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(sample.file).font(.subheadline)
                            Text(sample.reference)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                        Spacer()
                        Text(String(format: "%.1fs", sample.durationS))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .disabled(model.isBusy)
            }

            Button { importing = true } label: {
                Label("Choose an audio file", systemImage: "folder")
            }
            .disabled(model.isBusy)

            Button { Task { await model.toggleRecording() } } label: {
                Label(model.isRecording ? "Stop and transcribe" : "Record",
                      systemImage: model.isRecording ? "stop.circle.fill" : "mic.circle")
                    .foregroundStyle(model.isRecording ? .red : .accentColor)
            }
            .disabled(model.isBusy && !model.isRecording)
        }
    }

    private var liveSection: some View {
        Section("Live") {
            Button {
                Task { await model.toggleLive() }
            } label: {
                Label(model.isLive ? "Stop listening" : "Start live transcription",
                      systemImage: model.isLive ? "stop.circle.fill" : "waveform.badge.mic")
                    .foregroundStyle(model.isLive ? .red : .accentColor)
            }
            .disabled(model.isBusy && !model.isLive)

            if model.isLive {
                HStack(spacing: 8) {
                    Circle()
                        .fill(model.liveStatus == "Speaking" ? .green : .secondary)
                        .frame(width: 8, height: 8)
                    Text(model.liveStatus).font(.caption).foregroundStyle(.secondary)
                }
            }

            if !model.liveTranscript.isEmpty
                || !model.liveStable.isEmpty || !model.livePartial.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    if !model.liveTranscript.isEmpty {
                        Text(model.liveTranscript).textSelection(.enabled)
                    }
                    // Settled words in full colour, the draft tail dimmed, so
                    // it is obvious which part may still change.
                    if !model.liveStable.isEmpty || !model.livePartial.isEmpty {
                        // foregroundColor, not foregroundStyle: the latter is
                        // iOS 17+ and this target is 16.
                        (Text(model.liveStable).foregroundColor(.primary)
                         + Text(model.liveStable.isEmpty ? "" : " ")
                         + Text(model.livePartial).foregroundColor(.secondary))
                    }
                }
                .padding(.vertical, 2)
            }

            Toggle("Provisional results", isOn: $model.livePartialsEnabled)
                .font(.caption)
                .disabled(model.isLive)
        }
    }

    private var benchmarkSection: some View {
        Section("Benchmark") {
            if let progress = model.benchmarkProgress {
                HStack(spacing: 10) {
                    ProgressView()
                    Text(progress).font(.caption).foregroundStyle(.secondary)
                }
            } else if Benchmark.clips.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("No benchmark clips bundled").font(.subheadline)
                    Text("Run tools/fetch_benchmark_clips.py, then rebuild.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            } else {
                Button {
                    Task { await model.runBenchmark() }
                } label: {
                    Label("Run \(Benchmark.clips.count) clips x 2 decoders",
                          systemImage: "gauge.with.dots.needle.67percent")
                }
                .disabled(model.isBusy)
            }

            if !model.benchmarkRuns.isEmpty {
                VStack(spacing: 10) {
                    HStack {
                        Text("").frame(width: 54, alignment: .leading)
                        Text("WER").frame(maxWidth: .infinity, alignment: .trailing)
                        Text("median").frame(maxWidth: .infinity, alignment: .trailing)
                        Text("RTF").frame(maxWidth: .infinity, alignment: .trailing)
                        Text("time").frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)

                    ForEach(model.benchmarkRuns) { run in
                        HStack {
                            Text(run.decoder.rawValue.uppercased())
                                .font(.caption.weight(.semibold))
                                .frame(width: 54, alignment: .leading)
                            Text(String(format: "%.1f%%", run.meanWER * 100))
                                .frame(maxWidth: .infinity, alignment: .trailing)
                            Text(String(format: "%.1f%%", run.medianWER * 100))
                                .frame(maxWidth: .infinity, alignment: .trailing)
                            Text(String(format: "%.3f", run.realTimeFactor))
                                .frame(maxWidth: .infinity, alignment: .trailing)
                            Text(String(format: "%.0fs", run.processingSeconds))
                                .frame(maxWidth: .infinity, alignment: .trailing)
                        }
                        .font(.caption.monospacedDigit())
                    }

                    if let first = model.benchmarkRuns.first {
                        Text("\(first.clips) clips, \(Int(first.audioSeconds))s of \(model.language.name) audio")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    private var resultsSection: some View {
        Section("Results") {
            ForEach(model.outcomes) { outcome in
                VStack(alignment: .leading, spacing: 6) {
                    Text(outcome.text).textSelection(.enabled)
                    HStack(spacing: 8) {
                        Label(outcome.source, systemImage: "waveform")
                        Text(outcome.language.code.uppercased())
                        Text(outcome.decoder.rawValue.uppercased())
                        Text(String(format: "%.2fx", outcome.realTimeFactor))
                        if let wer = outcome.wordErrorRate {
                            Text(String(format: "WER %.1f%%", wer * 100))
                        }
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)
            }
        }
    }

    private var settingsSection: some View {
        Section("Settings") {
            Picker("Language", selection: $model.language) {
                ForEach(Language.allCases) { Text($0.name).tag($0) }
            }
            Picker("Decoder", selection: $model.decoder) {
                ForEach(SpeechRecognizer.Decoder.allCases) {
                    Text($0.rawValue.uppercased()).tag($0)
                }
            }
            .pickerStyle(.segmented)

            VStack(alignment: .leading, spacing: 4) {
                Text("Model host").font(.caption).foregroundStyle(.secondary)
                TextField("https://", text: $model.source)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .font(.caption.monospaced())
            }

            if model.cachedBytes > 0 {
                Button("Delete downloaded model", role: .destructive) {
                    Task { await model.deleteCache() }
                }
            }

            Text("Changing the language or decoder reloads the model. Switching "
                 + "language reuses the cached encoder and fetches only the new head.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isBusy)
    }
}

#Preview { ContentView() }
