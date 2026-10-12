import { writable } from 'svelte/store';
import { newTemperatureControlNetwork } from '$lib/networks/TemperatureControlNetwork';

function asyncReloadable<T>(loader: () => Promise<T>) {
	const { subscribe, set } = writable<T>(undefined as unknown as T);
	const load = async () => {
		const value = await loader();
		set(value);
	};
	return { subscribe, load, reload: load };
}

function persistable<T>(initialValue: T) {
	const { subscribe, set, update } = writable(initialValue);
	let current = initialValue;
	subscribe((v) => (current = v));
	return {
		subscribe,
		set,
		update,
		load: () => Promise.resolve(current)
	};
}

export const temperatureControlNetworkStore = asyncReloadable(async () =>
	newTemperatureControlNetwork()
);

export const learningRateStore = persistable(0.2);
