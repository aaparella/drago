package Drago

// Linear represents linear activation function
type Linear struct {
}

var _ Activator = new(Linear)

// Apply calculates activation for layer given previous layer value
func (l *Linear) Apply(r, c int, value float64) float64 {
	return value
}

// Derivative returns linear derivative (always 1)
func (l *Linear) Derivative(r, c int, value float64) float64 {
	return 1
}
