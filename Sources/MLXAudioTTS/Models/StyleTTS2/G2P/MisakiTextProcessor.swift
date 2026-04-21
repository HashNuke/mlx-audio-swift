import Foundation
import HuggingFace
import MLXAudioCore

public final class MisakiTextProcessor: TextProcessor, @unchecked Sendable {
    private static let requiredFilenames = [
        "us_gold.json",
        "us_silver.json",
        "us_bart_config.json",
        "us_bart.safetensors",
    ]

    private var usG2P: EnglishG2P?
    private var gbG2P: EnglishG2P?
    private let lock = NSLock()
    private let bundledResourceDirectory: URL?
    private nonisolated(unsafe) var resourceDirectory: URL?

    private static let g2pRepo = Repo.ID(namespace: "beshkenadze", name: "kitten-tts-g2p")

    public init() {
        self.bundledResourceDirectory = nil
    }

    public init(resourceDirectory: URL) {
        let standardizedURL = resourceDirectory.standardizedFileURL
        self.bundledResourceDirectory = standardizedURL
        self.resourceDirectory = standardizedURL
    }

    public func prepare() async throws {
        if let bundledResourceDirectory {
            try Self.validateResources(in: bundledResourceDirectory)
            lock.withLock {
                resourceDirectory = bundledResourceDirectory
            }
            return
        }

        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: Self.g2pRepo,
            requiredExtension: "safetensors"
        )
        lock.withLock {
            resourceDirectory = dir
        }
    }

    public func process(text: String, language: String?) throws -> String {
        let british = language?.lowercased().contains("gb") == true
        let g2p = try getG2P(british: british)
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }

    private static func validateResources(in directory: URL) throws {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw MisakiError.resourceDirectoryNotFound(directory)
        }

        let missingFiles = requiredFilenames.filter { filename in
            let fileURL = directory.appendingPathComponent(filename, isDirectory: false)
            return FileManager.default.fileExists(atPath: fileURL.path) == false
        }

        guard missingFiles.isEmpty else {
            throw MisakiError.missingResources(directory, missingFiles)
        }
    }

    private func getG2P(british: Bool) throws -> EnglishG2P {
        lock.lock()
        defer { lock.unlock() }
        guard let dir = resourceDirectory else {
            throw MisakiError.resourcesNotDownloaded
        }
        if british {
            if let cached = gbG2P { return cached }
            let g2p = try EnglishG2P(british: true, directory: dir)
            gbG2P = g2p
            return g2p
        } else {
            if let cached = usG2P { return cached }
            let g2p = try EnglishG2P(british: false, directory: dir)
            usG2P = g2p
            return g2p
        }
    }

    public enum MisakiError: Error, LocalizedError {
        case resourcesNotDownloaded
        case resourceDirectoryNotFound(URL)
        case missingResources(URL, [String])

        public var errorDescription: String? {
            switch self {
            case .resourcesNotDownloaded:
                return "G2P resources not downloaded. Call prepare() first."
            case .resourceDirectoryNotFound(let directory):
                return "Misaki resource directory not found at \(directory.path)"
            case .missingResources(let directory, let filenames):
                return "Misaki resources missing from \(directory.path): \(filenames.joined(separator: ", "))"
            }
        }
    }
}
