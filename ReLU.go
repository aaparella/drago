package Drago

import "math"

type ReLU struct {
}

func (u *ReLU) Apply(r, c int, val float64) float64 {
	return math.Log(1 + math.Exp(val))
}

func (u *ReLU) Derivative(r, c int, val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}
