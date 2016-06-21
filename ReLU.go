package drago

import "math"

// ReLU struct represents a rectified linear unit
type ReLU struct {
}

var _ Activator = new(ReLU)

// Apply ReLU calculation to val
func (u *ReLU) Apply(r, c int, val float64) float64 {
	return math.Log(1 + math.Exp(val))
}

// Derivative calculates derivative of ReLU for val
func (u *ReLU) Derivative(r, c int, val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}
