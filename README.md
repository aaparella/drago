Drago
===

Simple feed forward neural network implementation. Still need to add some nice utility functions, the logic can stand to be cleaned up in some places, but the algorithms are implemented and it can be used.

Usage:
===

    acts := Drago.Activator[]{new(Drago.Sigmoid), new(Drago.Sigmoid)}
    net := Drago.New(0.1, 25, []int{5, 2, 2, 1}, acts)
    net.Learn([][][]float64{
        {{0, 0}, {1}},
        {{0, 1}, {0}},
        {{1, 1}, {0}},
    })

    // Predict a value
    fmt.Println(net.Predict([]float64{1, 1})

To add an activation function: 
===

An activation function needs both the function and it's derivative. See Sigmoid.go, Tanh.go, and ReLU.go for examples of this.

    type YourActivationFunction struct {
    }

    func (y *YourActivationFunction) Apply(r, c int, val float64) float64 {
        // ...
    }

    func (y *YourActivationFunction) Derivative(r, c int, val float64) float64 {
        // ...
    }
