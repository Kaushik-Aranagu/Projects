import SwiftUI
import TesseraCore

// CLOUD-WRITTEN, NOT COMPILED. One day (sheet 243): the back word, the day's name in amber, the high in the day's
// color over the low in ink 3, its sentence, its hour columns, and the sun line. It slides in from the right (the
// room owns the .42 s spring); back is the word, or a swipe that starts in the 0-24 pt edge zone.

struct WeatherDayPage: View {
    let day: WeatherDay
    var calendar = Calendar.current
    let close: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                Button(action: close) {
                    Text("Weather")
                        .font(WeatherType.figtree(15, .medium, relativeTo: .subheadline))
                        .foregroundStyle(Theme.Colors.inkSecondary)
                        .frame(minWidth: 44, minHeight: 44, alignment: .leading)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Back to Weather")
                .accessibilityIdentifier("weather.back")
                Text(WeatherSentences.dayName(day.date, calendar: calendar))
                    .font(WeatherType.figtree(40, .semibold, relativeTo: .largeTitle))
                    .foregroundStyle(WeatherPalette.title)
                    .accessibilityAddTraits(.isHeader)
                    .accessibilityIdentifier("weather.day.title")
                HStack(alignment: .firstTextBaseline, spacing: Theme.Spacing.l) {
                    WeatherStepDown(size: 88) { size in
                        HStack(alignment: .firstTextBaseline, spacing: Theme.Spacing.l) {
                            Text("\(day.high)°")
                                .weatherNumeral(size)
                                .foregroundStyle(Theme.Colors.ink)
                            Text("\(day.low)°")
                                .weatherNumeral(size * 40 / 88)
                                .foregroundStyle(Theme.Colors.inkTertiary)
                        }
                        .lineLimit(1)
                    }
                }
                .padding(.top, Theme.Spacing.m)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("High \(day.high), low \(day.low)")
                .accessibilityIdentifier("weather.day.temps")
                WeatherSentenceText(sentence: WeatherSentences.daySentence(day, calendar: calendar), size: 24)
                    .padding(.top, Theme.Spacing.xl)
                WeatherHourStrip(hours: day.hours, firstHour: 6, calendar: calendar)
                    .frame(minHeight: 106, alignment: .topLeading)
                    .padding(.top, Theme.Spacing.xl)
                Text(WeatherSentences.sunLine(day, calendar: calendar))
                    .font(WeatherType.figtree(15, .regular, relativeTo: .subheadline))
                    .foregroundStyle(Theme.Colors.inkSecondary)
                    .padding(.top, Theme.Spacing.xl)
                    .accessibilityIdentifier("weather.day.sun")
            }
            .padding(.horizontal, Theme.Spacing.gutter)
            .padding(.top, Theme.Spacing.s)
            .padding(.bottom, Theme.Spacing.section)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .scrollIndicators(.hidden)
        .scrollBounceBehavior(.basedOnSize)
        .background { ThemedBackground() }
        .overlay(alignment: .leading) { edgeZone }
    }

    /// A little under the policy's 24 pt zone, so the first points of the back word (which starts at the gutter) stay
    /// tappable; the swipe itself still counts a start anywhere inside the 24 pt zone.
    private static let edgeWidth: CGFloat = 20

    /// The back-edge zone: a rightward swipe that starts here and goes `pageDistance` goes back.
    private var edgeZone: some View {
        Color.clear
            .frame(width: Self.edgeWidth)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: CGFloat(GestureTuning.dragStart), coordinateSpace: .global)
                    .onEnded { value in
                        let dx = Double(value.translation.width), dy = Double(value.translation.height)
                        let leads = GestureTuning.axis(dx: dx, dy: dy, bias: GestureTuning.pagerBias) == .horizontal
                        if leads, dx >= GestureTuning.pageDistance,
                           value.startLocation.x <= CGFloat(GestureTuning.backEdge)
                        {
                            close()
                        }
                    }
            )
            .accessibilityHidden(true)
    }
}
