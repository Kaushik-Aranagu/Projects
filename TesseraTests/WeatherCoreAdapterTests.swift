import Foundation
@testable import Tessera
import TesseraCore
import Testing

// CLOUD-WRITTEN, NOT COMPILED.
struct WeatherCoreAdapterTests {
    @Test func everyCoreConditionMapsToAKind() {
        let expected: [WeatherCondition: WeatherKind] = [
            .clear: .clear, .mainlyClear: .clear, .partlyCloudy: .partlyCloudy, .overcast: .cloudy, .fog: .fog,
            .drizzle: .rain, .freezingDrizzle: .rain, .rain: .rain, .freezingRain: .rain, .heavyRain: .heavyRain,
            .snow: .snow, .heavySnow: .snow, .snowGrains: .snow, .thunderstorm: .storm, .thunderstormHail: .storm,
        ]
        for condition in WeatherCondition.allCases {
            #expect(WeatherCoreAdapter.kind(condition) == expected[condition])
        }
    }

    @Test func windOnlyOverlaysADrySky() {
        #expect(WeatherCoreAdapter.kind(.overcast, windy: true) == .wind)
        #expect(WeatherCoreAdapter.kind(.rain, windy: true) == .rain)
        #expect(WeatherCoreAdapter.kind(.fog, windy: true) == .fog)
    }

    @Test func celsiusBecomesFahrenheitAndKphBecomesMph() {
        #expect(WeatherCoreAdapter.degrees(0) == 32)
        #expect(WeatherCoreAdapter.degrees(17) == 63)
    }

    @Test func aSampleReportBecomesAForecastWithNoPlace() {
        let report = SampleWeather.forecast(now: Date(timeIntervalSince1970: 1_800_000_000))
        let forecast = WeatherCoreAdapter.forecast(from: report)
        #expect(forecast.days.count == 7)
        #expect(forecast.hours.count == 48)
        #expect(forecast.temperature == 63)
        #expect(forecast.air.quality == "good")
        #expect(forecast.air.windFrom == nil)
        #expect(forecast.days.last?.hours.isEmpty == true)
    }

    @Test func oneWindRuleForNowHoursAndDays() {
        #expect(WeatherCoreAdapter.isWindy(windKph: 32, gustKph: 40))
        #expect(WeatherCoreAdapter.isWindy(windKph: 10, gustKph: 64))
        #expect(!WeatherCoreAdapter.isWindy(windKph: 31, gustKph: 63))
    }
}
