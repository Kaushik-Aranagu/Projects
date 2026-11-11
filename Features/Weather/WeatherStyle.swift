import SwiftUI

// CLOUD-WRITTEN, NOT COMPILED. Colors and type for the Weather room (sheet 243). Every condition wears one color in
// both rooms of the page. A page shows 3 or 4 colors: the amber title, the rain color, the dry slate, and ink.

extension WeatherKind {
    /// Two colors for the whole room (QA on #1401): the rain color for everything that falls, one calm slate for every
    /// other sky. The words keep their own effects; only the base colors collapse.
    var color: Color {
        isWet ? WeatherPalette.rain : WeatherPalette.dry
    }
}

enum WeatherPalette {
    /// The room's title: sky amber, the same as the sky word on Home.
    static var title: Color {
        Theme.Colors.Accent.skyFrom
    }

    /// The rain color: the Rain alert word and the rain words.
    static var rain: Color {
        Color(light: 0x557A93, dark: 0x9DBACE)
    }

    /// Every sky that does not fall: clear, cloudy, fog, wind.
    static var dry: Color {
        Color(light: 0x6F7C8A, dark: 0xA9B5C2)
    }
}

enum WeatherType {
    static func figtree(_ size: CGFloat, _ weight: Font.Weight = .regular,
                        relativeTo style: Font.TextStyle = .body) -> Font
    {
        Theme.Typography.face("Figtree", size: size, relativeTo: style, serif: false).weight(weight)
    }

    /// Outfit numerals: 200 from 28 pt up, 300 below, tabular (the house rule for figures).
    static func outfit(_ size: CGFloat, relativeTo style: Font.TextStyle = .body) -> Font {
        Theme.Typography.face("Outfit", size: size, relativeTo: style, serif: false)
            .weight(Theme.numeralWeight(size: size))
            .monospacedDigit()
    }
}

extension View {
    /// Figures carry the house tracking, -0.02 em.
    func weatherNumeral(_ size: CGFloat) -> some View {
        font(WeatherType.outfit(size, relativeTo: size > 40 ? .largeTitle : .title3))
            .tracking(-0.02 * size)
    }
}
