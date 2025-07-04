export const elu = `
fn activate(x: f32) -> f32 {
    return select(exp(x) - 1.0, x, x >= 0.0);
}`;