export const linear = `
fn activate(sum: f32, weightSum: f32) -> f32 {
    return sum;
}`;

export const elu = `
fn activate(sum: f32, weightSum: f32) -> f32 {
    let norm = sum / max(weightSum, 1e-5);
    return select(exp(norm) - 1.0, norm, norm >= 0.0);
}`;