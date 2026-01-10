use std::{error::Error, ffi::OsStr, fs::File, io::BufReader, path::PathBuf};

use burn::{
    module::AutodiffModule,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::backend::AutodiffBackend,
};
use zstd::Decoder;

use crate::dqn::{
    critic::CriticType,
    data_augmenter::DataAugmenterType,
    model::Model,
    replay_buffer::{ReplayBuffer, StateTransition},
    serialization::training_config::TrainingConfig,
    state::StateType,
    stats::StatsRecorderType,
    trainer::Trainer,
};

pub(crate) struct TrainingDeserializer {}

impl TrainingDeserializer {
    pub(crate) fn deserialize<B, M, S, C, R, D>(
        path: PathBuf,
        model: M,
    ) -> Result<(Trainer<B, M, S, C, R, D>, M), Box<dyn Error>>
    where
        B: AutodiffBackend,
        M: Model<B> + AutodiffModule<B>,
        M::InnerModule: Model<<B as AutodiffBackend>::InnerBackend>,
        S: StateType,
        C: CriticType<State = S>,
        R: StatsRecorderType<State = S>,
        D: DataAugmenterType<State = S>,
    {
        let config = TrainingDeserializer::deserialize_config(path.join("session.json"))?;
        let model = TrainingDeserializer::deserialize_model(path.join("model"), model)?;
        let replay_buffer: ReplayBuffer<S, D> =
            TrainingDeserializer::deserialize_replay_buffer(path.join("replay_buffer"), &config)?;
        let trainer = Trainer::from(config, replay_buffer, Default::default());
        Ok((trainer, model))
    }

    fn deserialize_config(path: PathBuf) -> Result<TrainingConfig, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|err| err.into())
    }

    fn deserialize_model<B: AutodiffBackend, M: Model<B> + AutodiffModule<B>>(
        file_path: PathBuf,
        model: M,
    ) -> Result<M, Box<dyn Error>> {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        let model = model
            .load_file(file_path, &recorder, &Default::default())
            .map_err(|err| err.into());

        model
    }

    fn deserialize_replay_buffer<S, D>(
        path: PathBuf,
        config: &TrainingConfig,
    ) -> Result<ReplayBuffer<S, D>, Box<dyn Error>>
    where
        S: StateType,
        D: DataAugmenterType<State = S>,
    {
        let files = get_chunk_list(&path)?;
        let mut transitions: Vec<StateTransition> =
            Vec::with_capacity(config.replay_buffer.capacity);
        for file_path in files.iter() {
            println!(
                "[dbg] Processing {}",
                file_path.file_name().unwrap().to_str().unwrap()
            );

            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            let decoder = Decoder::new(reader)?;
            let chunk: Vec<StateTransition> = bincode::deserialize_from(decoder)?;
            transitions.extend(chunk);
        }

        Ok(ReplayBuffer::from(transitions, &config.replay_buffer))
    }
}

fn get_chunk_list(path: &PathBuf) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut chunks = Vec::new();
    let extension = OsStr::new("chunk");
    for entry in path.read_dir()? {
        let entry = entry?;
        if entry.path().extension() == Some(extension) {
            chunks.push(entry.path());
        }
    }

    chunks.sort();
    Ok(chunks)
}
