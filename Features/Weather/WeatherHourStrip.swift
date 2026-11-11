import SwiftUI
import TesseraCore

// CLOUD-WRITTEN, NOT COMPILED. The hour columns (sheet 243): 59 pt wide, the time over the temperature over one word
// in the condition's color. A drag across (or up and down) pages three hours at a time; the columns slide, they do
// not snap. The drag goes through the app's gesture kit, so the axis, the distance and the flick are `GestureTuning`'s.

struct WeatherHourStrip: View {
    let hours: [WeatherHour]
    /// The first column shown, as an hour index into `hours`; it is rounded down to a whole page.
    var firstHour = 0
    var calendar = Calendar.current

    static let columnWidth: CGFloat = 59
    static let pageSize = 3
    static let height: CGFloat = 78

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var start: Int?

    var body: some View {
        GeometryReader { proxy in
            let visible = max(3, Int(proxy.size.width / Self.columnWidth))
            let last = max(0, hours.count - visible)
            let shown = min(start ?? (firstHour / Self.pageSize * Self.pageSize), last)
            HStack(alignment: .top, spacing: 0) {
                ForEach(Array(hours.enumerated()), id: \.element.id) { index, hour in
                    column(hour, index: index)
                }
            }
            .offset(x: -CGFloat(shown) * Self.columnWidth)
            .frame(width: proxy.size.width, alignment: .leading)
            .clipped()
            .contentShape(Rectangle())
            // The strip owns a drag that starts on it, across or up and down, so it never also scrolls the page.
            // (It has no tap targets, so a high-priority drag swallows nothing.)
            .highPriorityGesture(
                DragGesture(minimumDistance: CGFloat(GestureTuning.dragStart), coordinateSpace: .global)
                    .onEnded { value in turn(value, from: shown, last: last) }
            )
        }
        .frame(height: Self.height)
        .accessibilityIdentifier("weather.hours")
    }

    private func turn(_ value: DragGesture.Value, from shown: Int, last: Int) {
        let dx = Double(value.translation.width), dy = Double(value.translation.height)
        let across = GestureTuning.pageTurn(
            startX: Double(value.startLocation.x), dx: dx, dy: dy, velocityX: Double(value.velocity.width)
        )
        if across != 0 {
            return move(across > 0 ? 1 : -1, from: shown, last: last)
        }
        let pull = GestureTuning.pull(dx: dx, dy: dy, velocityY: Double(value.velocity.height))
        if pull != 0 {
            move(pull > 0 ? -1 : 1, from: shown, last: last)
        }
    }

    private func move(_ pages: Int, from shown: Int, last: Int) {
        let next = min(max(shown + pages * Self.pageSize, 0), last)
        guard next != shown else { return }
        // Reduce Motion cuts to the next page.
        guard !reduceMotion else { return start = next }
        withTesseraAnimation(Theme.Motion.house, reduceMotion: false) { start = next }
    }

    private func column(_ hour: WeatherHour, index: Int) -> some View {
        VStack(spacing: Theme.Spacing.xs) {
            Text(WeatherSentences.columnTime(hour.time, calendar: calendar))
                .font(WeatherType.figtree(13, .regular, relativeTo: .footnote))
                .foregroundStyle(Theme.Colors.inkSecondary)
            Text("\(hour.temperature)°")
                .weatherNumeral(24)
                .foregroundStyle(Theme.Colors.ink)
            Text(hour.kind.short)
                .font(WeatherType.figtree(13, .semibold, relativeTo: .footnote))
                .foregroundStyle(hour.kind.color)
        }
        .lineLimit(1)
        .frame(width: Self.columnWidth)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            "\(WeatherSentences.columnTime(hour.time, calendar: calendar)), \(hour.temperature) degrees, "
                + hour.kind.short
        )
        .accessibilityIdentifier("weather.hour.\(index)")
    }
}
