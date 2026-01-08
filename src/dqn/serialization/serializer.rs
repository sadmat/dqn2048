use std::{
    error::Error,
    fmt::{self, write},
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::PathBuf,
};

// use core::error::Error;

use burn::{
    module::{AutodiffModule, Module},
    prelude::Backend,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::backend::AutodiffBackend,
};
use zstd::stream::{AutoFinishEncoder, write::Encoder};

use crate::dqn::{
    critic::CriticType,
    data_augmenter::DataAugmenterType,
    model::Model,
    replay_buffer::{self, ReplayBuffer},
    serialization::training_config::{ReplayBufferConfig, TrainingConfig, TrainingInfo},
    state::StateType,
    stats::StatsRecorderType,
    trainer::{Hyperparameters, Trainer},
};

#[derive(Debug)]
enum SessionSerializationError {
    PathNotEmpty,
}

impl Error for SessionSerializationError {}
impl fmt::Display for SessionSerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write(f, format_args!("{:?}", self))
    }
}

struct TrainingSerializer {}

impl TrainingSerializer {
    fn serialize<B, M, S, C, R, D>(
        trainer: &Trainer<B, M, S, C, R, D>,
        model: M,
        path: PathBuf,
    ) -> Result<(), Box<dyn Error>>
    where
        B: AutodiffBackend,
        M: Model<B> + AutodiffModule<B>,
        M::InnerModule: Model<<B as AutodiffBackend>::InnerBackend>,
        S: StateType,
        C: CriticType<State = S>,
        R: StatsRecorderType<State = S>,
        D: DataAugmenterType<State = S>,
    {
        if !is_path_empty(&path)? {
            return Err(SessionSerializationError::PathNotEmpty.into());
        }
        TrainingSerializer::serialize_config(trainer, &path)?;
        TrainingSerializer::serialize_model(model, path.join("model"))?;
        TrainingSerializer::serialize_replay_buffer(trainer.replay_buffer(), &path);
        Ok(())
    }

    fn serialize_config<B, M, S, C, R, D>(
        trainer: &Trainer<B, M, S, C, R, D>,
        path: &PathBuf,
    ) -> Result<(), Box<dyn Error>>
    where
        B: AutodiffBackend,
        M: Model<B> + AutodiffModule<B>,
        M::InnerModule: Model<<B as AutodiffBackend>::InnerBackend>,
        S: StateType,
        C: CriticType<State = S>,
        R: StatsRecorderType<State = S>,
        D: DataAugmenterType<State = S>,
    {
        let config = TrainingConfig::with(
            trainer.hyperparameters().clone(),
            ReplayBufferConfig::with(
                trainer.replay_buffer().capacity(),
                trainer.replay_buffer().size(),
                trainer.replay_buffer().write_position(),
            ),
            TrainingInfo::with(trainer.epoch_number(), trainer.frame_number()),
        );

        let file_path = path.join("session.json");
        let file = File::create(file_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &config)?;

        Ok(())
    }

    fn serialize_model<B: Backend, M: Model<B> + Module<B>>(
        model: M,
        file_path: PathBuf,
    ) -> Result<(), Box<dyn Error>> {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        model
            .save_file(file_path, &recorder)
            .map_err(|err| err.into())
    }

    fn serialize_replay_buffer<S, D>(
        replay_buffer: &ReplayBuffer<S, D>,
        path: &PathBuf,
    ) -> Result<(), Box<dyn Error>>
    where
        S: StateType,
        D: DataAugmenterType<State = S>,
    {
        let replay_buffer_dir = path.join("replay_buffer");
        let chunk_size = 2_u32.pow(16) as usize;
        let mut chunk_start = 0;
        let mut chunk_number = 1;

        fs::create_dir(replay_buffer_dir.clone())?;

        while chunk_start < replay_buffer.size() {
            println!(
                "[dbg] Processing chunk #{chunk_number} [{chunk_start}..{}]",
                chunk_start + chunk_size
            );
            let chunk = replay_buffer.transitions(chunk_start, chunk_start + chunk_size);
            let file_path = replay_buffer_dir.join(format!("{:05}.chunk", chunk_number));
            let file = File::create(file_path)?;
            let encoder = Encoder::new(file, 3)?.auto_finish();
            bincode::serialize_into(encoder, chunk)?;

            chunk_start += chunk_size;
            chunk_number += 1;
        }

        Ok(())
    }
}

fn is_path_empty(path: &PathBuf) -> Result<bool, Box<dyn Error>> {
    path.read_dir()
        .and_then(|mut entry| Ok(entry.next().is_none()))
        .map_err(|err| err.into())
}
