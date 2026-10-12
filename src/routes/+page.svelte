<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { temperatureControlNetworkStore } from '../stores';
	import * as tslog from 'tslog';
	import { allLinks } from '$lib/LinkFilters';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import {
		computeApparentTemperature,
		MAX_TEMPERATURE,
		MAX_WATER_VAPORT_PRESSURE,
		MAX_WIND_SPEED,
		MIN_TEMPERATURE,
		MIN_WATER_VAPOR_PRESSURE,
		MIN_WIND_SPEED,
		setApparentTemperatureFunctionWeights
	} from '$lib/networks/TemperatureControlNetwork';

	const logger = new tslog.Logger({ name: 'temperature' });

	const networkStore = temperatureControlNetworkStore;

	let _tick = $state(0);

	const networkShape = $derived($networkStore[0]?.shape);
	const weights = $derived.by(() => {
		void _tick;
		const w = $networkStore[0]?.tfModel.weights;
		return w ? [...w] : undefined;
	});
	const neuronFormula = $derived(buildNeuronFormula(weights));
	const currentStats = $derived.by(() => {
		void _tick;
		const s = $networkStore[0]?.stats;
		return s ? { ...s, losses: [...s.losses] } : null;
	});

	const formulaFormatter = Intl.NumberFormat('en', {
		notation: 'compact',
		maximumFractionDigits: 2
	});

	function buildNeuronFormula(tfWeights: tf.LayerVariable[] | undefined) {
		// Build something like:  1,0 * T - 0.19 * V + 0.33 * H - 4
		if (!tfWeights) {
			return;
		}
		const withOperator = (x: number) =>
			x > 0 ? `+ ${formulaFormatter.format(x)}` : `- ${formulaFormatter.format(Math.abs(x))}`;
		const [weights, biases] = tfWeights.map((w) => w.read().flatten().arraySync() as number[]);
		const [wT, wV, wH] = weights;
		const bias = biases[0];
		return `${formulaFormatter.format(wT)} * T ${withOperator(wV)} * V ${withOperator(wH)} * H ${withOperator(bias)}`;
	}

	const formatter = Intl.NumberFormat('en', { maximumFractionDigits: 1 });

	let temperature = $state(MIN_TEMPERATURE);
	let windSpeed = $state(MIN_WIND_SPEED);
	let waterVaporPressure = $state(MIN_WATER_VAPOR_PRESSURE);

	let temperatureBlocked = $state(false);
	let pressureBlocked = $state(false);
	let windBlocked = $state(false);

	let prediction: number[] | undefined = $state(undefined);
	let activations: number[][] | undefined = $state(undefined);
	let linkFilter = allLinks;

	const formattedPrediction = $derived(formatter.format(prediction as any));
	const formattedComputedApparentTemperature = $derived(
		formatter.format(computeApparentTemperature(temperature, windSpeed, waterVaporPressure))
	);

	let isLoading = $state(true);

	onMount(async () => {
		await networkStore.load();
		isLoading = false;
		// setApparentTemperatureFunctionWeights($networkStore.tfModel);
		predict_apparent_temperature();
	});

	const randomUniformInt = (min: number, max: number) =>
		Math.floor(Math.random() * (max - min) + min);

	function show_example_in_UI() {
		console.log("Blocked ? " + temperatureBlocked);
		
		if ( !temperatureBlocked )
			temperature = randomUniformInt(MIN_TEMPERATURE, MAX_TEMPERATURE);
		if ( !windBlocked )
			windSpeed = randomUniformInt(MIN_WIND_SPEED, MAX_WIND_SPEED);
		if ( !pressureBlocked )
			waterVaporPressure = randomUniformInt(MIN_WATER_VAPOR_PRESSURE, MAX_WATER_VAPORT_PRESSURE);
		predict_apparent_temperature();
	}

	function predict_apparent_temperature() {
		const result = calculateActivations(temperature, windSpeed, waterVaporPressure);
		activations = result;
		prediction = result[result.length - 1];
	}

	function calculateActivations(
		temperature: number,
		windSpeed: number,
		waterVaporPressure: number
	): number[][] {
		return tf.tidy(() => {
			const input = tf.tensor([temperature, windSpeed, waterVaporPressure]).reshape([-1, 3]);
			const rawPrediction = $networkStore[0].featureModel.predict(input);
			// predict() returns Tensor when there is a single output layer, Tensor[] otherwise
			const activationTensors: tf.Tensor[] = Array.isArray(rawPrediction)
				? rawPrediction
				: [rawPrediction as tf.Tensor];
			const layerActivations = activationTensors.map((x) => {
				const values = tf.squeeze(x).arraySync();
				// A single-unit layer produces a scalar; wrap it so every layer has number[]
				return (Array.isArray(values) ? values : [values]) as number[];
			});
			const inputActivations = tf.squeeze(input).arraySync() as number[];
			return [inputActivations, ...layerActivations];
		});
	}

	async function trainOnData(trainDataSize: number) {
		const trainData = $networkStore[0].nextTrainBatch(trainDataSize);
		train(trainData.xs, trainData.ys, 32);
	}

	async function train(trainXs: tf.Tensor2D, trainYs: tf.Tensor2D, batchSize: number) {
		const networkUnderTraining = $networkStore[0];

		function onBatchEnd(batch: number, logs: tf.Logs = {}) {
			logger.debug('end batch:', batch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: logs.size,
				finalAccuracy: logs.acc,
				loss: logs.loss
			});
			_tick++; // Notify subscribers
			show_example_in_UI();
		}

		function onEpochEnd(epoch: number, logs: tf.Logs = {}) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
		}

		function onTrainEnd(_logs: tf.Logs = {}) {
			logger.debug('onTrain end : tf.memory()', tf.memory());
			// tf.dispose(trainXs);
			// tf.dispose(trainYs); // For this network they live forever. XXX
			logger.debug('after disposing: tf.memory()', tf.memory());

			predict_apparent_temperature();
		}

		logger.debug('Before fit: tf.memory()', tf.memory());

		// If this fails because there is already another fit running
		// Then the 2 tensors get leaked (because the cleanup occurs in
		// onTrainEnd, which is never called)
		const params: tf.ModelFitArgs = {
			epochs: 1,
			batchSize: batchSize,
			shuffle: false,
			callbacks: { onBatchEnd, onEpochEnd, onTrainEnd } as tf.CustomCallbackArgs
		};
		return networkUnderTraining.tfModel.fit(trainXs, trainYs, params);
	}

	async function train1() {
		trainOnData(1);
	}

	async function train100() {
		trainOnData(100);
	}

	async function train1000() {
		trainOnData(1000);
	}

	function resetModel() {
		networkStore.reload();
		_tick++; // Notify subscribers
		predict_apparent_temperature();
	}

	function setATFunctionWeights() {
		const networkUnderTraining = $networkStore[0];
		setApparentTemperatureFunctionWeights(networkUnderTraining.tfModel);
		_tick++; // Notify subscribers
		predict_apparent_temperature();
	}

	function blockTemperature() {
		temperatureBlocked = !temperatureBlocked;
		if ( temperatureBlocked )
			$networkStore[1].setFixedTemperature(temperature)
		else
			$networkStore[1].resetTemperatureInterval();
	}

	function blockWind() {
		windBlocked = !windBlocked
		if ( windBlocked )
			$networkStore[1].setFixedWindSpeed(windSpeed)
		else
			$networkStore[1].resetWindSpeedInterval();
	}

	function blockPressure() {
		pressureBlocked = !pressureBlocked
		if ( pressureBlocked )
			$networkStore[1].setFixedWaterVaporPressure(waterVaporPressure)
		else
			$networkStore[1].resetWaterVaporPressureInterval();
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-6 gap-9">
		<!-- Left -->
		<div class="col-span-1 shadow p-2">
			<h1 class="text-2xl mt-3 mb-6">Apprentissage</h1>
			<button class="btn btn-outline btn-primary mt-5 mb-5 p-6" onclick={train1}>
				Entraîner avec 1 exemple
			</button>
			<button class="btn btn-outline btn-primary mt-5 mb-5 p-6" onclick={train100}>
				Entraîner avec 100 exemples
			</button>
			<button class="btn btn-outline btn-primary mt-5 mb-5 p-6" onclick={train1000}>
				Entraîner avec 1000 exemples
			</button>
		</div>

		<!-- Center -->
		<div class="col-span-3">
			<!-- Controls -->
			<div class="grid grid-cols-3 stats stats-horizontal">
				<div class="ml-2 mr-2 text-center">
					<h2 class="text-xl mt-1 mb-1">Température</h2>
					<p class="stat-title !text-base">(en °C)</p>
					<input
						type="range"
						min={MIN_TEMPERATURE}
						max={MAX_TEMPERATURE}
						bind:value={temperature}
						oninput={predict_apparent_temperature}
						class="range range-primary range-xs"
					/>
					<p>
					Fixer la valeur
					<input
						type="checkbox"
						checked={temperatureBlocked}
						onchange={blockTemperature}
					/>
					</p>
					<!-- <div class="stat-value !text-xl">{temperature} °C</div> -->
				</div>
				<div class="ml-2 mr-2 text-center">
					<h2 class="text-xl mt-1 mb-1">Vent</h2>
					<p class="stat-title !text-base">(en km/h)</p>
					<input
						type="range"
						min={MIN_WIND_SPEED}
						max={MAX_WIND_SPEED}
						bind:value={windSpeed}
						oninput={predict_apparent_temperature}
						class="range range-primary range-xs"
					/>
					<p>
					Fixer la valeur
					<input
						type="checkbox"
						checked={windBlocked}
						onchange={blockWind}
					/>
					</p>
					<!-- <div class="stat-value !text-xl">{windSpeed} km/h</div> -->
				</div>
				<div class="ml-2 mr-2 text-center">
					<h2 class="text-xl mt-1 mb-1">Pression atmosphérique</h2>
					<p class="stat-title !text-base">(en hPa)</p>
					<input
						type="range"
						min={MIN_WATER_VAPOR_PRESSURE}
						max={MAX_WATER_VAPORT_PRESSURE}
						bind:value={waterVaporPressure}
						oninput={predict_apparent_temperature}
						class="range range-primary range-xs"
					/>
					<p>
					Fixer la valeur
					<input
						type="checkbox"
						checked={pressureBlocked}
						onchange={blockPressure}
					/>
					</p>
					<!-- <div class="stat-value !text-xl">{waterVaporPressure} hPa</div> -->
				</div>
			</div>
			<!-- Network -->
			<div>
				<NetworkGraph
					{networkShape}
					weights={weights ?? []}
					activations={activations ?? []}
					{linkFilter}
					style=""
				/>
			</div>
			<!-- Output -->
			<div class="grid grid-cols-3 stats stats-horizontal">
				<div class=""></div>
				<div class="text-center">
					<p class="stat-value text-2xl">{formattedPrediction} °C</p>
					<h2 class="text-xl mt-1 mb-1">Température ressentie calculée</h2>
					<p class="stat-title !text-base">(inférée)</p>
					<!-- <p class="stat-title !text-base">Formule actuelle :</p>
					<p class="stat-value text-base">{neuronFormula}</p> -->
				</div>
				<div class="text-center">
					<p class="stat-value !text-2xl">{formattedComputedApparentTemperature} °C</p>
					<h2 class="text-xl mt-1 mb-1">Température ressentie mesurée</h2>
					<p class="stat-title !text-base">(réelle)</p>
				</div>
			</div>
		</div>

		<div class="row align-items-center">
			<NetworkStats stats={currentStats!} />
			<button class="btn btn-outline btn-error mb-5 mt-5 align-center" onclick={resetModel}>
				Réinitialiser le réseau
			</button>
			<button class="btn btn-outline btn-primary" onclick={setATFunctionWeights}>
				Réseau cible
			</button>
		</div>
	</div>
{/if}
