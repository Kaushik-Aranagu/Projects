import SwiftUI

// CLOUD-WRITTEN, NOT COMPILED. A big figure that gives way one point at a time, down to three quarters of its size,
// when its line runs out of room (never a smooth squeeze). Past that it is allowed to wrap or cut like any text.

struct WeatherStepDown<Content: View>: View {
    let size: CGFloat
    @ViewBuilder let content: (CGFloat) -> Content

    private var steps: [CGFloat] {
        let floor = (size * 0.75).rounded(.up)
        return Array(stride(from: size, through: floor, by: -1))
    }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            ForEach(steps, id: \.self) { step in
                content(step).fixedSize(horizontal: true, vertical: false)
            }
        }
    }
}
