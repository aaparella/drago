package Drago

import "math"

// Tanh struct represents hyperbolic tangent activation function
type Tanh struct {
}

var _ Activator = new(Tanh)

// Apply calculates hyperbolic tangent of val, r and c ignored
func (t *Tanh) Apply(r, c int, val float64) float64 {
	return (1 - math.Exp(-2*val)) / (1 + math.Exp(-2*val))
}

// Derivative calculates derivative of hyperbolic tangent for val, r and c ignored
func (t *Tanh) Derivative(r, c int, val float64) float64 {
	return 1 - (math.Pow((math.Exp(2*val)-1)/(math.Exp(2*val)+1), 2))
}
