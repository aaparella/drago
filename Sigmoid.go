package drago

import "math"

// Sigmoid represents sigmoid activation function
type Sigmoid struct {
}

var _ Activator = new(Sigmoid)

// Apply calculates sigmoid of given value, r and c ignored
func (s *Sigmoid) Apply(r, c int, value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

// Derivative calculates sigmoid derivative of given value, r and c ignored
func (s *Sigmoid) Derivative(r, c int, value float64) float64 {
	res := s.Apply(r, c, value)
	return res * (1 - res)
}
