package Drago

type Activator interface {
	Apply(float64) float64
	Derivative(float64) float64
}
