<script lang="ts">
	import '../app.css';
	import Logo from './_Logo.svelte';

	let { children } = $props();
</script>

<div class="navbar px-8 py-0">
	<Logo size={40} />
	<p class="text-lg mx-1">Un neurone</p>
</div>

<div class="divider m-0 pb-6"></div>

<div class="container px-8">
	{@render children()}
</div>
