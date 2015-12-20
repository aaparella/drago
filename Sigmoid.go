package Drago

import "math"

type Sigmoid struct {
}

func (s *Sigmoid) Apply(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

func (s *Sigmoid) Derivative(value float64) float64 {
	res := s.Apply(value)
	return res * (1 - res)
}
