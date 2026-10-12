<script lang="ts">
	import { type LayerVariable } from '@tensorflow/tfjs';
	import { type DenseNetwork, Link, Neuron, type Layer } from '$lib/NetworkShape';
	import type { LinkFilter } from '$lib/LinkFilters';
	import { DefaultMap, zip2 } from '$lib/generic/utils';
	import { hsv2rgb } from '$lib/generic/image';
	import plotly from 'plotly.js-dist';
	import { visualWeight } from '$lib/LinkFilters';

	let {
		networkShape,
		activations,
		weights,
		linkFilter,
		style,
		onNeuronSelected
	}: {
		networkShape: DenseNetwork;
		activations: number[][];
		weights: LayerVariable[];
		linkFilter: LinkFilter;
		style: string;
		onNeuronSelected?: (neuron: Neuron | null) => void;
	} = $props();

	let plotElement: HTMLElement;
	let plotlyInitialized = true;

	let neuronColor = "#C7E1FF";
	let weightColor = '#eeeeee';
	let defaultLineStyle = { color: 'black', width: 1 };
	let neuronLinkStyle = { color: 'black', width: 4 };
	let rectangleLinkStyle = { color: 'black', width: 1 };

	$effect(() => {
		drawGraph(networkShape, activations, weights, linkFilter);
		if (!plotlyInitialized) {
			plotlyInitialized = true;
			(plotElement as any).on('plotly_hover', onPlotlyHover);
			(plotElement as any).on('plotly_unhover', onPlotlyUnhover);
		}
	});

	function lookup_neuron(curveNumber: number, pointNumber: number): Neuron | null {
		return networkShape.layers[curveNumber]?.neurons[pointNumber];
	}

	function onPlotlyHover(data: any) {
		if (data.points) {
			const point = data.points[0];
			const neuron = lookup_neuron(point.curveNumber, point.pointNumber);
			onNeuronSelected?.(neuron);
		}
	}

	function onPlotlyUnhover(_: any) {
		onNeuronSelected?.(null);
	}

	function neuronTraces(networkShape: DenseNetwork, activations: (number[] | null)[] | null) {
		const traces = [];
		const lastNeuron = networkShape.layers[1].neurons[0];
		for (const [i, layer] of networkShape.layers.entries()) {
			const layerActivations = activations?.[i];
			const trace = {
				x: layer.neurons.map((n) => n.x),
				y: layer.neurons.map((n) => n.y),
				name: layer.name,
				mode: 'markers',
				marker: {
					// Without array.from mapping doesn't work properly
					color: neuronColor,
					symbol: 'circle',
					line: { width: 1 },
					size: 50
				}
			};
			traces.push(trace);
			if ( i==0 ) {
				// Ajout des formes au milieu des liaisons
				const trace = {
					x: layer.neurons.map((n) => (n.x+lastNeuron.x)/2),
					y: layer.neurons.map((n) => (n.y+lastNeuron.y)/2),
					mode: 'markers',
					marker: {
						// Without array.from mapping doesn't work properly
						color: weightColor,
						symbol: 'pentagon',
						line: { width: 1 },
						size: 85
					}
				};
				traces.push(trace);
				;
				// Ajout du neurone de biais
				trace.x.push( (layer.neurons[layer.neurons.length-1].x+lastNeuron.x)/2+1 )
				trace.y.push( (layer.neurons[layer.neurons.length-1].y+lastNeuron.y)/2 )
			}
		}
		return traces;
	}

	function link_color(weight: number): string {
		// Red and blue ramps depending on the sign
		const [r, g, b] = hsv2rgb(weight > 0 ? 0.66 : 0, Math.abs(weight), 0.9);
		return `rgb(${r},${g},${b})`;
	}

	function link_width(weight: number): number {
		return Math.abs(weight * 1.5);
	}

	function linkTraces(links: Link[], lineStyle : object = defaultLineStyle) {
		// To draw each line exactly how we want it we need to return one scatter-plot per line,
		// and displaying that is super slow.
		// Instead we group the lines into buckets and draw all the lines in the same bucket
		// with the same color and width and make that a single scatter-plot.
		// By varying the number of buckets one can balance precision of display with rendering speed.

		const bucket_size = 20; // the number of buckets is the range of weights times the bucket_size

		// Each bucket contains a list of coordinates with gaps (nulls).
		// See the gaps section of:  https://plotly.com/python/line-charts/
		const edge_x_buckets = new DefaultMap<number, any>(() => []);
		const edge_y_buckets = new DefaultMap<number, any>(() => []);
		for (const link of links) {
			const linkVisualWeight = visualWeight(link);
			const bucket = Math.round(linkVisualWeight * bucket_size);
			const edge_x = edge_x_buckets.get(bucket);
			const edge_y = edge_y_buckets.get(bucket);
			edge_x.push(link.a.x);
			edge_y.push(link.a.y);
			edge_x.push(link.b.x);
			edge_y.push(link.b.y);
			edge_x.push(null);
			edge_y.push(null);
		}

		const traces = [];
		for (const bucket of edge_x_buckets.keys()) {
			const trace = {
				x: edge_x_buckets.get(bucket),
				y: edge_y_buckets.get(bucket),
				mode: 'lines',
				hoverinfo: 'skip',
				line: lineStyle
			};
			traces.push(trace);
		}

		return traces;
	}

	const defaultGraphLayout = {
		xaxis: {
			visible: false,
			uirevision: 'time' // preserve zoom and pan between redraws
		},
		yaxis: {
			visible: false,
			scaleanchor: 'x', // 1:1 aspect ratio
			uirevision: 'time' // preserve zoom and pan between redraws
		},
		showlegend: false,
		//font: { size: 18, color: 'black' },
		font: { size: 20, color: 'black' },
		margin: { t: 0, l: 0, r: 0, b: 0 }
	};

	const graphConfig = {
		displayModeBar: false,
		scrollZoom: false,
		responsive: false, // Important, otherwise anything under it in the layout floats back up
		staticPlot: true
	};

	// https://plotly.com/javascript/text-and-annotations/
	type Annotation = {
		x: number;
		y: number;
		// yanchor: string;
		// yshift: number;
		// textangle: number;
		text: string; // Can be html
		showarrow: boolean;
	};

	const formulaFormatter = Intl.NumberFormat('en', {
		notation: 'compact',
		maximumFractionDigits: 2
	});

	function buildAnnotations(layers: Layer[], weights: LayerVariable[]): Annotation[] {
		const annotations : Annotation[] = [];
		const lastNeuron = networkShape.layers[1].neurons[0];

		const [w, biases] = weights.map((w) => w.read().flatten().arraySync() as number[]);

		let weightIndex = 0;
		for (const [i, layer] of layers.entries()) {
			if (layer.classes) {
				// let maxActivation = undefined;
				// if (layer == layers[layers.length - 1]) {
				// 	maxActivation = Math.max(...layer.neurons.map((n) => n.activation));
				// }
				for (const [neuron, class_] of zip2(layer.neurons, layer.classes)) {
					const circle : Annotation = {
						x: neuron.x,
						y: neuron.y,
						//yanchor: rank == 0 ? 'bottom' : 'top',
						//textangle: layer.rotateClassNames ? -90 : 0,
						//yshift: rank == 0 ? 10 : -10,
						text: i==0 ? ''+neuron.activation : '<span style="font-style: bold; font-size: 3rem;">+</span>',
							// neuron.activation > 0 && neuron.activation == maxActivation
							// 	? '<b>' + class_ + '</b>'
							// 	: class_,
						showarrow: false
					};
					annotations.push(circle);
					// Ajout du texte au milieu des liaisons
					if ( i==0 ) {
						const texte : Annotation = {
							x: (neuron.x+lastNeuron.x)/2,
							y: (neuron.y+lastNeuron.y)/2,
							text: "&#215;<br>"+formulaFormatter.format(w[weightIndex]),
							showarrow: false
						};
						annotations.push(texte);
						weightIndex++;
					}
				}
				if ( i==0 ) {
					const bias : Annotation = {
						x: (layer.neurons[2].x+lastNeuron.x)/2+1,
						y: (layer.neurons[2].y+lastNeuron.y)/2,
						text: formulaFormatter.format(biases[0]),
						showarrow: false
					};
					annotations.push(bias);
				}
			}
		}
		return annotations;
	}

	function drawGraph(
		networkShape: DenseNetwork,
		activations: number[][],
		weights: LayerVariable[] | undefined,
		linkFilter: LinkFilter
	) {
		if (!plotElement) {
			console.log('no plotElement');
			return;
		}
		const traces = [];

		// 2. Links
		if (weights) {
			const weightsBetweenLayers = weights.filter((w) => w.originalName.endsWith('kernel'));
			const links = networkShape.getLinks(weightsBetweenLayers, activations);
			const outputNeuron = networkShape.layers[1].neurons[0];

			traces.push(...linkTraces(linkFilter(links),neuronLinkStyle));

			// Trace un rectangle autour du neurone
			let neuronRectangle = [
				new Link(
					new Neuron((networkShape.layers[0].neurons[0].x+outputNeuron.x)/2-0.5,(networkShape.layers[0].neurons[2].y+outputNeuron.y)/2+0.6),
					new Neuron((networkShape.layers[0].neurons[2].x+outputNeuron.x)/2+1.5,(networkShape.layers[0].neurons[2].y+outputNeuron.y)/2+0.6),
					0
				),
				new Link(
					new Neuron((networkShape.layers[0].neurons[0].x+outputNeuron.x)/2-0.5,(networkShape.layers[0].neurons[0].y+outputNeuron.y)/2+0.6),
					new Neuron((networkShape.layers[0].neurons[0].x+outputNeuron.x)/2-0.5,outputNeuron.y-0.35),
					0
				),
				new Link(
					new Neuron((networkShape.layers[0].neurons[2].x+outputNeuron.x)/2+1.5,(networkShape.layers[0].neurons[2].y+outputNeuron.y)/2+0.6),
					new Neuron((networkShape.layers[0].neurons[2].x+outputNeuron.x)/2+1.5,outputNeuron.y-0.35),
					0
				),
				new Link(
					new Neuron((networkShape.layers[0].neurons[2].x+outputNeuron.x)/2+1.5,outputNeuron.y-0.35),
					new Neuron((networkShape.layers[0].neurons[0].x+outputNeuron.x)/2-0.5,outputNeuron.y-0.35),
					0
				),
			];

			traces.push(...linkTraces(neuronRectangle,rectangleLinkStyle));


			links.push(
				// Ajout du lien depuis le biais vers le neurone de sortie
				new Link(
					new Neuron((networkShape.layers[0].neurons[2].x+outputNeuron.x)/2+1,(networkShape.layers[0].neurons[2].y+outputNeuron.y)/2),
					new Neuron(outputNeuron.x,outputNeuron.y),
					0
				),
				// Ajout du lien qui sort du neurone de sortie
				new Link(
					new Neuron(outputNeuron.x,outputNeuron.y),
					new Neuron(outputNeuron.x,outputNeuron.y-0.7),
					0
				),
				// Ajout du lien qui sort du neurone de sortie (flèche à gauche)
				new Link(
					new Neuron(outputNeuron.x-0.1,outputNeuron.y-0.5),
					new Neuron(outputNeuron.x,outputNeuron.y-0.7),
					0
				),
				// Ajout du lien qui sort du neurone de sortie (flèche à droite)
				new Link(
					new Neuron(outputNeuron.x+0.1,outputNeuron.y-0.5),
					new Neuron(outputNeuron.x,outputNeuron.y-0.7),
					0
				)
			);
			traces.push(...linkTraces(linkFilter(links),neuronLinkStyle));
		}

		// 1. Neurons
		if (activations) {
			traces.push(...neuronTraces(networkShape, activations));
		} else if (weights) {
			let biasesTensors = weights.filter((w) => w.originalName.endsWith('bias'));
			const biases: (number[] | null)[] = biasesTensors.map(
				(t) => t.read().arraySync() as number[]
			);
			biases.unshift(null); // The input layer does not have biases
			traces.push(...neuronTraces(networkShape, biases));
		}

		// 3. Annotations
		const graphLayout = {
			...defaultGraphLayout,
			annotations: buildAnnotations(networkShape.layers,weights?weights:[]),
		};

		plotly.react('network-graph', traces, graphLayout, graphConfig);
	}
</script>

<div bind:this={plotElement} id="network-graph" class="network-graph" {style}></div>
