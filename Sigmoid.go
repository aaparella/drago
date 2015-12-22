package Drago

import "math"

type Sigmoid struct {
}

var _ Activator = new(Sigmoid)

func (s *Sigmoid) Apply(r, c int, value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

func (s *Sigmoid) Derivative(r, c int, value float64) float64 {
	res := s.Apply(r, c, value)
	return res * (1 - res)
}
