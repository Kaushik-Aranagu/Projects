import Foundation
@testable import Tessera
import Testing

// CLOUD-WRITTEN, NOT COMPILED.
struct WeatherSentencesTests {
    private var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .current
        return calendar
    }

    private var forecast: WeatherForecast {
        SampleWeatherSource.make(now: Date(timeIntervalSince1970: 1_800_000_000), calendar: calendar)
    }

    @Test func theHeadlineNamesTheSkyNowAndTheFirstChange() {
        #expect(WeatherSentences.headline(forecast, calendar: calendar).text == "Cloudy now, rain from 4 PM.")
    }

    @Test func theDetailNamesTheHeavyHoursAndWhenTheyEnd() {
        let detail = WeatherSentences.detail(forecast, calendar: calendar)
        #expect(detail?.text == "Heavy rain after 6 PM, clearing by midnight.")
    }

    @Test func theWeekNamesOnlyTheNotableDaysAndEachIsADoor() throws {
        let week = WeatherSentences.week(forecast, calendar: calendar)
        let doors = week.pieces.compactMap { piece -> (WeatherKind, Int)? in
            if case let .word(kind, _, day?) = piece { (kind, day) } else { nil }
        }
        #expect(doors.map(\.0) == [.rain, .wind])
        #expect(doors.map(\.1) == [2, 4])
        #expect(week.text.hasSuffix("."))
    }

    @Test func aQuietWeekSaysSo() {
        let quiet = WeatherForecast(
            now: forecast.now, temperature: 60, feelsLike: 60, kind: .clear, high: 62, low: 50, hours: forecast.hours,
            days: forecast.days.map {
                WeatherDay(date: $0.date, high: $0.high, low: $0.low, kind: .clear, sunrise: $0.sunrise,
                           sunset: $0.sunset, hours: $0.hours)
            },
            air: forecast.air
        )
        #expect(WeatherSentences.week(quiet, calendar: calendar).text == "A quiet week, nothing to plan around.")
    }

    @Test func clocksReadInTheHouseShortForm() {
        let date = calendar.date(from: DateComponents(year: 2027, month: 1, day: 5, hour: 7, minute: 25)) ?? Date()
        #expect(WeatherSentences.clock(date, calendar: calendar) == "7:25 AM")
        #expect(WeatherSentences.columnTime(date, calendar: calendar) == "7 AM")
    }

    @Test func noForecastTypeCarriesAPlace() {
        // The seam has no city, coordinates or place: a forecast is only the sky.
        let names = Mirror(reflecting: forecast).children.compactMap(\.label)
        #expect(!names.contains { ["city", "place", "location", "latitude", "longitude"].contains($0) })
    }
}
