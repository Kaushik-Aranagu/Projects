import Foundation
import TesseraCore

/// The weight unit Tess speaks in, for the few places that have no store (the grammar's asks). Set once at launch and
/// when Settings > Units changes, like `QuietHours.current`; stored values never change, only the words in her asks.
public enum TessUnits {
    public nonisolated(unsafe) static var weight: MassUnit = .kilograms

    /// "kg" or "lb".
    public static var symbol: String {
        weight.symbol
    }
}
