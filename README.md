# Krause Synchronization Transformers
This repository contains the implementation for the paper Krause Synchronization Transformers. In our work, we introduce <strong>Krause Attention</strong>, a principled attention mechanism inspired by bounded-confidence consensus dynamics. Krause Attention replaces similarity-based global aggregation with distance-based, localized, and selectively sparse interactions, promoting structured local synchronization instead of global mixing. We relate this behavior to recent theory modeling Transformer dynamics as interacting particle systems, and show how bounded-confidence interactions naturally moderate attention concentration and alleviate attention sinks. Restricting interactions to local neighborhoods also reduces runtime complexity from quadratic to linear in sequence length. Experiments across vision (ViT on CIFAR/ImageNet), autoregressive generation (MNIST/CIFAR-10), and large language models (Llama/Qwen) demonstrate consistent gains with substantially reduced computation, highlighting bounded-confidence dynamics as a scalable and effective inductive bias for attention.

<section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body">
      <div class="has-text-centered">
        <img src="images/figure1.svg" 
             alt="Description of the new attention mechanism" 
             style="width: 80%; height: auto; display: inline-block;"> 
             </div>
    </div>
  </div>
</section>

<section class="hero is-small">
  <div class="hero-body" style="background: transparent;">
    <div class="container">
      <div class="item" style="max-width: 850px; margin: 0 auto;">
        <h2 class="title is-2.5 has-text-left">Krause Attention</h2>
        <div class="has-text-centered">
          <img src="images/kst_gif.gif" alt="Teaser GIF" style="width: 100%; height: auto; display: block;">
        </h2>
      </div>
    </div>
  </div>
</section>

<section class="hero is-small">
  <div class="hero-body" style="background-color: #f5f5f5 !important; padding: 40px 0;">
    <div class="container">
      <div class="item" style="max-width: 850px; margin: 0 auto;">
        <h2 class="title is-2.5 has-text-left">Alleviating Attention Sinks in Krause-LLMs</h2>
        <div class="has-text-centered">
          <img src="images/attention_sink_llama3.svg" alt="Second research result visualization" loading="lazy" style="max-width: 100%; height: auto; display: block;"/>
        </div>
        </h2>
      </div>
    </div>
  </div>
</section>

<section class="hero is-small" style="background: transparent;">
  <div class="hero-body">
    <div class="container">
      <div class="item" style="max-width: 850px; margin: 0 auto;">        
        <h2 class="title is-2.5 has-text-left">Attention Heatmaps in Vision Transformers</h2>        
        <div class="has-text-centered">
          <img src="images/imagenet_heatmap_main.svg" 
               alt="Attention Heatmaps" 
               loading="lazy" 
               style="max-width: 100%; height: auto; display: block;"/>
        </div>
        <div class="has-text-centered">
          <img src="images/attention_evolution_map.svg" 
               alt="Attention Evolution" 
               loading="lazy" 
               style="max-width: 100%; height: auto; display: block;"/>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="hero is-small" style="background-color: #f5f5f5 !important; padding: 40px 0;">
  <div class="hero-body">
    <div class="container">
      <div class="item" style="max-width: 1000px; margin: 0 auto;">
        <h2 class="title is-2.5 has-text-centered">Krause Autoregressive Transformers for Image Generation</h2>        
        <div class="columns is-vcentered is-variable is-5">
          <div class="column">
            <div class="has-text-centered">
              <img src="images/completion_mnist.svg" 
                   alt="Attention Heatmaps" 
                   loading="lazy" 
                   style="width: 100%; height: auto; display: block;"/>
            </div>
          </div>
          <div class="column">
            <div class="has-text-centered">
              <img src="images/completion_cifar10.svg" 
                   alt="Attention Evolution Across Layers" 
                   loading="lazy" 
                   style="width: 100%; height: auto; display: block;"/>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>



