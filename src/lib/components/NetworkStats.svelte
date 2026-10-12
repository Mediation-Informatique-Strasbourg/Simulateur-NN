<script lang="ts">
	import type { NetworkStats } from '$lib/Network';
	import LossChart from '$lib/components/LossChart.svelte';

	let { stats }: { stats: NetworkStats } = $props();

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	const formattedNumExamples = $derived(formatter.format(stats.samplesSeen));
	const losses = $derived(stats.losses??[]);
</script>

<div class="stats bg-base-200 stats-vertical">
	<div class="stat">
		<div class="stat-title !text-base text-center">Exemples vus</div>
		<div class="stat-value text-center">{formattedNumExamples}</div>
	</div>
	{#if losses?.length}
		<div class="stat">
			<div class="stat-title !text-base text-center">Erreur</div>
			<div class="stat-value text-center">{formatter.format(losses[losses.length - 1].loss)}</div>
			<LossChart {losses} />
		</div>
	{/if}
</div>
