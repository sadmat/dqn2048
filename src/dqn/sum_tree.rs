pub(crate) struct SumTree {
    capacity: usize,
    storage: Vec<f32>,
    last_index: usize,
}

impl SumTree {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "Sum tree capacity has to be a power of 2"
        );
        Self {
            capacity,
            storage: vec![0.0; 2 * capacity],
            last_index: 0,
        }
    }

    pub(crate) fn total(&self) -> f32 {
        self.storage[1]
    }

    pub(crate) fn update(&mut self, mut index: usize, value: f32) {
        assert!(index < self.capacity, "Given index is out of bounds");
        self.last_index = self.last_index.max(index);
        index += self.capacity;
        self.storage[index] = value;

        while index > 1 {
            index /= 2;
            self.storage[index] = self.left(index) + self.right(index);
        }
    }

    pub(crate) fn sample(&self, value: f32) -> (usize, f32) {
        let index = self.sample_index(value);
        (index, self.storage[index + self.capacity])
    }

    pub(crate) fn sample_index(&self, mut value: f32) -> usize {
        let mut index = 1;

        while index < self.capacity {
            if value < self.left(index) {
                index = self.left_index(index);
            } else {
                value -= self.left(index);
                index = self.right_index(index);
            }
        }

        (index - self.capacity).min(self.last_index)
    }

    fn left(&self, index: usize) -> f32 {
        self.storage[self.left_index(index)]
    }

    fn right(&self, index: usize) -> f32 {
        self.storage[self.right_index(index)]
    }

    fn left_index(&self, index: usize) -> usize {
        index * 2
    }

    fn right_index(&self, index: usize) -> usize {
        index * 2 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_sum_is_zero() {
        let tree = SumTree::with_capacity(8);
        assert_eq!(tree.total(), 0.0);
    }

    #[test]
    fn value_update_changes_total_sum() {
        let mut tree = SumTree::with_capacity(8);
        tree.update(0, 1.0);
        assert_eq!(tree.total(), 1.0);
        tree.update(1, 1.0);
        assert_eq!(tree.total(), 2.0);
        tree.update(2, 1.0);
        assert_eq!(tree.total(), 3.0);
    }

    #[test]
    fn sampling() {
        let mut tree = SumTree::with_capacity(8);
        for i in 0..8 {
            tree.update(i, i as f32 + 1.0);
        }

        let mut value = 0.0;
        for i in 0..8 {
            value += i as f32;
            assert_eq!(
                tree.sample_index(value),
                i,
                "Value {value} should be mapped to index {i}"
            );
        }
    }
}
