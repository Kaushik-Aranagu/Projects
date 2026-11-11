import SwiftUI
import TesseraCore

// CLOUD-WRITTEN, NOT COMPILED. The Weather room (sheet 243, board v1): the landing and, over it, one Day page. Fixed
// slots, no cards, pills or bars; the title is sky amber, the temperature wears the condition, and the foot is one
// action word in the rain color. The forecast comes through `WeatherForecastSource`, so Core's Open-Meteo source
// plugs in without touching this file. No city is ever drawn, and nothing here is written to a backup, a log or a
// widget: the rain alert is one on/off preference on this phone.

struct WeatherRoomView: View {
    var source: any WeatherForecastSource = SampleWeatherSource()
    /// Core schedules (or cancels) the 30-minute rain notification when the word is tapped.
    var onRainAlert: (Bool) -> Void = { _ in }
    var calendar = Calendar.current

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var forecast: WeatherForecast?
    @State private var loaded = false
    @State private var openDay: Int?
    @State private var rainAlert = UserDefaults.standard.bool(forKey: WeatherAlertDefaults.key)
    @State private var receipt: Receipt?

    struct Receipt: Equatable {
        let text: String
        let id = UUID()
    }

    var body: some View {
        ZStack {
            landing
            if let forecast, let openDay, forecast.days.indices.contains(openDay) {
                WeatherDayPage(day: forecast.days[openDay], calendar: calendar, close: closeDay)
                    .transition(reduceMotion ? .identity : .move(edge: .trailing))
                    .zIndex(1)
            }
        }
        .background { ThemedBackground() }
        .task {
            forecast = await source.forecast()
            loaded = true
        }
    }

    /// Reduce Motion cuts: the page is just there, with no fade and no slide.
    private func show(_ index: Int) {
        guard !reduceMotion else { return openDay = index }
        withTesseraAnimation(Theme.Motion.house, reduceMotion: false) { openDay = index }
    }

    private func closeDay() {
        guard !reduceMotion else { return openDay = nil }
        withTesseraAnimation(Theme.Motion.house, reduceMotion: false) { openDay = nil }
    }

    // MARK: Landing

    private var landing: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                if let forecast {
                    content(forecast)
                } else {
                    header(time: nil)
                    if loaded {
                        Text("No forecast yet.")
                            .font(WeatherType.figtree(17, .regular, relativeTo: .body))
                            .foregroundStyle(Theme.Colors.inkSecondary)
                            .padding(.top, Theme.Spacing.xl)
                            .accessibilityIdentifier("weather.empty")
                    }
                }
            }
            .padding(.horizontal, Theme.Spacing.gutter)
            .padding(.top, Theme.Spacing.s)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .scrollIndicators(.hidden)
        .scrollBounceBehavior(.basedOnSize)
        .safeAreaInset(edge: .bottom, spacing: 0) { foot }
        .accessibilityHidden(openDay != nil)
    }

    @ViewBuilder
    private func content(_ forecast: WeatherForecast) -> some View {
        header(time: WeatherSentences.clock(forecast.now, calendar: calendar))
        temperature(forecast)
            .frame(minHeight: 126, alignment: .topLeading)
        WeatherSentenceText(sentence: WeatherSentences.headline(forecast, calendar: calendar), size: 26)
            .frame(minHeight: 38, alignment: .topLeading)
            .accessibilityIdentifier("weather.headline")
        Group {
            if let detail = WeatherSentences.detail(forecast, calendar: calendar) {
                WeatherSentenceText(sentence: detail, size: 17, weight: .regular, ink: Theme.Colors.inkSecondary)
            }
        }
        .frame(minHeight: 54, alignment: .topLeading)
        label("Next hours")
        WeatherHourStrip(hours: forecast.hours, calendar: calendar)
            .frame(minHeight: 106, alignment: .topLeading)
        label("The week")
        WeatherDoorSentence(sentence: WeatherSentences.week(forecast, calendar: calendar), size: 22,
                            lineHeight: 34, open: show)
            .frame(minHeight: 164, alignment: .topLeading)
        airLine(forecast.air)
            .padding(.bottom, Theme.Spacing.xl)
    }

    private func header(time: String?) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Weather")
                .font(WeatherType.figtree(40, .semibold, relativeTo: .largeTitle))
                .foregroundStyle(WeatherPalette.title)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: Theme.Spacing.m)
            if let time {
                Text(time)
                    .weatherNumeral(15)
                    .foregroundStyle(Theme.Colors.inkSecondary)
            }
        }
        .frame(minHeight: 58, alignment: .topLeading)
    }

    private func temperature(_ forecast: WeatherForecast) -> some View {
        HStack(alignment: .top, spacing: Theme.Spacing.l) {
            WeatherStepDown(size: 104) { size in
                HStack(alignment: .top, spacing: 2) {
                    Text("\(forecast.temperature)")
                        .weatherNumeral(size)
                    Text("°")
                        .weatherNumeral(size / 2)
                        .padding(.top, size * 0.08)
                }
                .lineLimit(1)
            }
            .foregroundStyle(Theme.Colors.ink)
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                Text("feels like \(forecast.feelsLike)°")
                Text("high \(forecast.high)°, low \(forecast.low)°")
            }
            .font(WeatherType.figtree(15, .regular, relativeTo: .subheadline))
            .foregroundStyle(Theme.Colors.inkSecondary)
            .padding(.top, 34)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(forecast.temperature) degrees, feels like \(forecast.feelsLike), "
            + "high \(forecast.high), low \(forecast.low)")
        .accessibilityIdentifier("weather.temperature")
    }

    private func label(_ text: String) -> some View {
        Text(text)
            .font(WeatherType.figtree(13, .medium, relativeTo: .footnote))
            .foregroundStyle(Theme.Colors.inkSecondary)
            .frame(minHeight: 20, alignment: .topLeading)
            .accessibilityAddTraits(.isHeader)
    }

    private func airLine(_ air: WeatherAir) -> some View {
        let line = WeatherSentences.airLine(air)
        var text = AttributedString(line.before)
        let quality = AttributedString(line.quality)
        text += quality
        text += AttributedString(line.after)
        return Text(text)
            .font(WeatherType.figtree(15, .regular, relativeTo: .subheadline))
            .foregroundStyle(Theme.Colors.inkSecondary)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("weather.air")
    }

    // MARK: Foot

    private var foot: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let receipt {
                HStack(spacing: Theme.Spacing.m) {
                    Text(receipt.text)
                        .font(WeatherType.figtree(15, .regular, relativeTo: .subheadline))
                        .foregroundStyle(Theme.Colors.ink)
                        .accessibilityIdentifier("weather.receipt")
                    Button(action: undo) {
                        Text("Undo")
                            .font(WeatherType.figtree(15, .semibold, relativeTo: .subheadline))
                            .foregroundStyle(WeatherPalette.rain)
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("weather.undo")
                }
                .transition(.identity)
            }
            HStack(alignment: .center) {
                Button(action: toggleAlert) {
                    HStack(spacing: Theme.Spacing.xs) {
                        Text("Rain alert")
                            .font(WeatherType.figtree(19, .semibold, relativeTo: .body))
                            .foregroundStyle(WeatherPalette.rain)
                        if rainAlert {
                            Text("on")
                                .font(WeatherType.figtree(15, .regular, relativeTo: .subheadline))
                                .foregroundStyle(Theme.Colors.inkSecondary)
                        }
                    }
                    .frame(minHeight: 44)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityValue(rainAlert ? "On" : "Off")
                .accessibilityHint("Toggles a note 30 minutes before rain")
                .accessibilityIdentifier("weather.rainAlert")
                Spacer(minLength: Theme.Spacing.m)
                // Tess sits here, on the right (her layer draws over this slot).
                Color.clear.frame(width: 64, height: 44)
                    .accessibilityHidden(true)
            }
        }
        // The action word starts at least `actionInset` from the left edge: the gutter plus what is missing.
        .padding(.leading, Theme.Spacing.gutter + max(0, CGFloat(GestureTuning.actionInset) - Theme.Spacing.gutter))
        .padding(.trailing, Theme.Spacing.gutter)
        .padding(.bottom, Theme.Spacing.s)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func toggleAlert() {
        setAlert(!rainAlert, text: !rainAlert ? "You'll hear 30 minutes before rain." : "Rain alert is off.")
    }

    private func setAlert(_ on: Bool, text: String?) {
        rainAlert = on
        UserDefaults.standard.set(on, forKey: WeatherAlertDefaults.key)
        onRainAlert(on)
        guard let text else {
            receipt = nil
            return
        }
        let shown = Receipt(text: text)
        receipt = shown
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(GestureTuning.undoSeconds))
            if receipt?.id == shown.id { receipt = nil }
        }
    }

    private func undo() {
        setAlert(!rainAlert, text: nil)
    }
}
