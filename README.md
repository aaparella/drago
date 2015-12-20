Drago
===

Simple feed forward neural network implementation. Doesn't actually work yet, just as a heads up you know.

Usage:

    acts := Drago.Activator[]{new(Drago.Sigmoid), new(Drago.Sigmoid)}
    net := Drago.New(0.1, 25, []int{5, 2, 2, 1}, acts)
    net.Learn([][][]float64{
        {{0, 0}, {1}},
        {{0, 1}, {0}},
        {{1, 1}, {0}},
    })
