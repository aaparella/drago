package Drago

type Linear struct {
}

var _ Activator = new(Linear)

func (l *Linear) Apply(r, c int, value float64) float64 {
	return value
}

func (l *Linear) Derivative(r, c int, value float64) float64 {
	return 1
}
