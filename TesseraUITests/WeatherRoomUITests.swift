import XCTest

// CLOUD-WRITTEN, NOT COMPILED. The Weather room's taps and swipes (gesture policy G11-G13): every tap target is
// tapped, the hour columns page by swipe, and the Day page goes back by its word and by the edge swipe.
@MainActor
final class WeatherRoomUITests: SimulatorOnlyTestCase {
    private func launch() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-demo", "-demoTime", "10:00", "-openLink", "weather"]
        app.launch()
        let alert = app.descendants(matching: .any)["weather.rainAlert"]
        XCTAssertTrue(alert.waitForExistence(timeout: 30), "the Weather room opens")
        return app
    }

    private func element(_ app: XCUIApplication, _ id: String) -> XCUIElement {
        app.descendants(matching: .any)[id].firstMatch
    }

    func testTheRainAlertWordTogglesWithAReceiptAndUndo() {
        let app = launch()
        element(app, "weather.rainAlert").tap()
        let receipt = element(app, "weather.receipt")
        XCTAssertTrue(receipt.waitForExistence(timeout: 3))
        XCTAssertEqual(receipt.label, "You'll hear 30 minutes before rain.")
        element(app, "weather.undo").tap()
        XCTAssertFalse(element(app, "weather.receipt").waitForExistence(timeout: 1), "Undo clears the receipt")
        XCTAssertEqual(element(app, "weather.rainAlert").value as? String, "Off", "Undo turned the alert back off")
    }

    func testAWeatherWordInTheWeekOpensItsDayAndBackWordReturns() {
        let app = launch()
        let door = element(app, "weather.door.2")
        XCTAssertTrue(door.waitForExistence(timeout: 5), "a weather word in The week")
        door.tap()
        XCTAssertTrue(element(app, "weather.day.title").waitForExistence(timeout: 3), "the Day page")
        element(app, "weather.back").tap()
        XCTAssertTrue(element(app, "weather.rainAlert").waitForExistence(timeout: 3), "back on the landing")
    }

    func testTheEdgeSwipeGoesBackFromADay() {
        let app = launch()
        element(app, "weather.door.2").tap()
        XCTAssertTrue(element(app, "weather.day.title").waitForExistence(timeout: 3))
        let edge = app.coordinate(withNormalizedOffset: CGVector(dx: 0.01, dy: 0.5))
        edge.press(forDuration: 0.1, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.7, dy: 0.5)))
        XCTAssertTrue(element(app, "weather.rainAlert").waitForExistence(timeout: 3), "the edge swipe went back")
    }

    func testTheHourColumnsPageByASwipeAcross() {
        let app = launch()
        let strip = element(app, "weather.hours")
        XCTAssertTrue(strip.waitForExistence(timeout: 5))
        let first = element(app, "weather.hour.0")
        let before = first.frame.minX
        let from = strip.coordinate(withNormalizedOffset: CGVector(dx: 0.85, dy: 0.5))
        from.press(forDuration: 0.1, thenDragTo: strip.coordinate(withNormalizedOffset: CGVector(dx: 0.15, dy: 0.5)))
        sleep(1)
        XCTAssertLessThan(first.frame.minX, before - 100, "the columns slid three hours")
    }

    func testAVerticalDragOnTheStripPagesTheHoursAndLeavesTheScrollAlone() {
        let app = launch()
        let strip = element(app, "weather.hours")
        XCTAssertTrue(strip.waitForExistence(timeout: 5))
        let temperatureTop = element(app, "weather.temperature").frame.minY
        let first = element(app, "weather.hour.0")
        let before = first.frame.minX
        let from = strip.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.8))
        from.press(forDuration: 0.1, thenDragTo: strip.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.0)))
        sleep(1)
        XCTAssertLessThan(first.frame.minX, before - 100, "an upward drag on the strip paged the hours")
        XCTAssertEqual(element(app, "weather.temperature").frame.minY, temperatureTop, accuracy: 1,
                       "the room did not scroll")
    }

    func testAVerticalDragOffTheStripStillScrollsTheRoom() {
        let app = XCUIApplication()
        app.launchArguments = ["-demo", "-demoTime", "10:00", "-openLink", "weather",
                               "-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXL"]
        app.launch()
        let headline = element(app, "weather.headline")
        XCTAssertTrue(headline.waitForExistence(timeout: 30))
        let before = element(app, "weather.temperature").frame.minY
        let start = headline.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        start.press(forDuration: 0.1, thenDragTo: start.withOffset(CGVector(dx: 0, dy: -200)))
        sleep(1)
        XCTAssertLessThan(element(app, "weather.temperature").frame.minY, before - 50, "a drag off the strip scrolled")
    }
}
