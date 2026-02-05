# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Flask application for arena visualization."""

import io
import os

from flask import Flask, jsonify, render_template, request, send_file, abort, redirect

from genarena.visualize.data_loader import ArenaDataLoader


def create_app(arena_dir: str, data_dir: str) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        arena_dir: Path to arena directory
        data_dir: Path to data directory

    Returns:
        Configured Flask app
    """
    # Get the directory containing this file for templates/static
    app_dir = os.path.dirname(os.path.abspath(__file__))

    app = Flask(
        __name__,
        template_folder=os.path.join(app_dir, "templates"),
        static_folder=os.path.join(app_dir, "static"),
    )

    # Store paths in config
    app.config["ARENA_DIR"] = arena_dir
    app.config["DATA_DIR"] = data_dir

    # Create data loader
    data_loader = ArenaDataLoader(arena_dir, data_dir)

    # ========== Page Routes ==========

    @app.route("/")
    def index():
        """Main page."""
        return render_template("index.html")

    # ========== API Routes ==========

    @app.route("/api/subsets")
    def api_subsets():
        """Get list of available subsets."""
        subsets = data_loader.discover_subsets()
        return jsonify({"subsets": subsets})

    @app.route("/api/subsets/<subset>/info")
    def api_subset_info(subset: str):
        """Get information about a subset."""
        info = data_loader.get_subset_info(subset)
        if not info:
            return jsonify({"error": "Subset not found"}), 404

        return jsonify({
            "name": info.name,
            "models": info.models,
            "experiments": info.experiments,
            "total_battles": info.total_battles,
            "min_input_images": info.min_input_images,
            "max_input_images": info.max_input_images,
            "prompt_sources": info.prompt_sources,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/battles")
    def api_battles(subset: str, exp_name: str):
        """Get paginated battle records."""
        # Parse query parameters
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 20, type=int)
        result_filter = request.args.get("result", None, type=str)
        consistency = request.args.get("consistent", None, type=str)
        min_images = request.args.get("min_images", None, type=int)
        max_images = request.args.get("max_images", None, type=int)
        prompt_source = request.args.get("prompt_source", None, type=str)

        # Support multiple models (comma-separated or multiple params)
        models_param = request.args.get("models", None, type=str)
        models = None
        if models_param:
            models = [m.strip() for m in models_param.split(",") if m.strip()]

        # Convert consistency filter
        consistency_filter = None
        if consistency == "true":
            consistency_filter = True
        elif consistency == "false":
            consistency_filter = False

        # Get battles
        records, total = data_loader.get_battles(
            subset=subset,
            exp_name=exp_name,
            page=page,
            page_size=page_size,
            models=models,
            result_filter=result_filter,
            consistency_filter=consistency_filter,
            min_images=min_images,
            max_images=max_images,
            prompt_source=prompt_source,
        )

        return jsonify({
            "battles": [r.to_dict() for r in records],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/battles/<path:battle_id>")
    def api_battle_detail(subset: str, exp_name: str, battle_id: str):
        """Get detailed battle record."""
        # Parse battle_id: model_a_vs_model_b:sample_index
        try:
            parts = battle_id.rsplit(":", 1)
            sample_index = int(parts[1])
            model_part = parts[0]

            # Split model names
            if "_vs_" in model_part:
                models = model_part.split("_vs_")
                model_a, model_b = models[0], models[1]
            else:
                return jsonify({"error": "Invalid battle_id format"}), 400
        except (ValueError, IndexError):
            return jsonify({"error": "Invalid battle_id format"}), 400

        record = data_loader.get_battle_detail(
            subset, exp_name, model_a, model_b, sample_index
        )

        if not record:
            return jsonify({"error": "Battle not found"}), 404

        return jsonify(record.to_detail_dict())

    @app.route("/api/subsets/<subset>/stats")
    def api_stats(subset: str):
        """Get statistics for a subset."""
        exp_name = request.args.get("exp_name", None, type=str)
        stats = data_loader.get_stats(subset, exp_name)

        if not stats:
            return jsonify({"error": "Subset not found"}), 404

        return jsonify(stats)

    @app.route("/api/subsets/<subset>/leaderboard")
    def api_elo_leaderboard(subset: str):
        """Get ELO leaderboard for a subset."""
        # Support multiple models filter (comma-separated)
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        leaderboard = data_loader.get_elo_leaderboard(subset, filter_models)
        return jsonify({"leaderboard": leaderboard})

    @app.route("/api/subsets/<subset>/models/<path:model>/stats")
    def api_model_stats(subset: str, model: str):
        """Get detailed statistics for a specific model including win rates against all opponents."""
        exp_name = request.args.get("exp_name", "__all__", type=str)
        stats = data_loader.get_model_vs_stats(subset, model, exp_name)

        if not stats:
            return jsonify({"error": "Model not found"}), 404

        return jsonify(stats)

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/h2h")
    def api_head_to_head(subset: str, exp_name: str):
        """Get head-to-head statistics between two models."""
        model_a = request.args.get("model_a", None, type=str)
        model_b = request.args.get("model_b", None, type=str)

        if not model_a or not model_b:
            return jsonify({"error": "model_a and model_b are required"}), 400

        h2h = data_loader.get_head_to_head(subset, exp_name, model_a, model_b)
        return jsonify(h2h)

    @app.route("/api/subsets/<subset>/samples/<int:sample_index>/input_count")
    def api_input_image_count(subset: str, sample_index: int):
        """Get the number of input images for a sample."""
        count = data_loader.get_input_image_count(subset, sample_index)
        return jsonify({"count": count})

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/samples/<int:sample_index>/all_models")
    def api_sample_all_models(subset: str, exp_name: str, sample_index: int):
        """Get all model outputs for a specific sample, sorted by win rate."""
        # Support multiple models filter (comma-separated)
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        # stats_scope: 'filtered' = only count battles between filtered models
        #              'all' = count all battles (but show only filtered models)
        stats_scope = request.args.get("stats_scope", "filtered", type=str)

        result = data_loader.get_sample_all_models(
            subset, exp_name, sample_index, filter_models, stats_scope
        )

        if not result:
            return jsonify({"error": "Sample not found"}), 404

        return jsonify(result)

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/samples/<int:sample_index>/models/<path:model>/battles")
    def api_model_battles_for_sample(subset: str, exp_name: str, sample_index: int, model: str):
        """Get all battle records for a specific model on a specific sample."""
        # Parse optional opponent models filter (comma-separated)
        opponents_param = request.args.get("opponents", None, type=str)
        opponent_models = None
        if opponents_param:
            opponent_models = [m.strip() for m in opponents_param.split(",") if m.strip()]

        result = data_loader.get_model_battles_for_sample(
            subset=subset,
            exp_name=exp_name,
            sample_index=sample_index,
            model=model,
            opponent_models=opponent_models,
        )

        return jsonify(result)

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/prompts")
    def api_prompts(subset: str, exp_name: str):
        """Get paginated list of prompts/samples with all model outputs."""
        # Parse query parameters
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 10, type=int)
        min_images = request.args.get("min_images", None, type=int)
        max_images = request.args.get("max_images", None, type=int)
        prompt_source = request.args.get("prompt_source", None, type=str)

        # Support multiple models filter (comma-separated)
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        # Get prompts
        prompts, total = data_loader.get_prompts(
            subset=subset,
            exp_name=exp_name,
            page=page,
            page_size=page_size,
            min_images=min_images,
            max_images=max_images,
            prompt_source=prompt_source,
            filter_models=filter_models,
        )

        return jsonify({
            "prompts": prompts,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/search")
    def api_search(subset: str, exp_name: str):
        """Search battles by text query (full-text search across instruction, task_type, prompt_source, metadata)."""
        # Parse query parameters
        query = request.args.get("q", "", type=str)
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 20, type=int)
        consistency = request.args.get("consistent", None, type=str)

        # Support multiple models (comma-separated)
        models_param = request.args.get("models", None, type=str)
        models = None
        if models_param:
            models = [m.strip() for m in models_param.split(",") if m.strip()]

        # Convert consistency filter
        consistency_filter = None
        if consistency == "true":
            consistency_filter = True
        elif consistency == "false":
            consistency_filter = False

        # Search battles
        records, total = data_loader.search_battles(
            subset=subset,
            exp_name=exp_name,
            query=query,
            page=page,
            page_size=page_size,
            models=models,
            consistency_filter=consistency_filter,
        )

        return jsonify({
            "battles": [r.to_dict() for r in records],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "query": query,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/search/prompts")
    def api_search_prompts(subset: str, exp_name: str):
        """Search prompts by text query."""
        # Parse query parameters
        query = request.args.get("q", "", type=str)
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 10, type=int)

        # Support multiple models filter (comma-separated)
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        # Search prompts
        prompts, total = data_loader.search_prompts(
            subset=subset,
            exp_name=exp_name,
            query=query,
            page=page,
            page_size=page_size,
            filter_models=filter_models,
        )

        return jsonify({
            "prompts": prompts,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "query": query,
        })

    @app.route("/api/subsets/<subset>/matrix")
    def api_win_rate_matrix(subset: str):
        """Get win rate matrix for all model pairs."""
        exp_name = request.args.get("exp_name", "__all__", type=str)

        # Support model filter (comma-separated)
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        result = data_loader.get_win_rate_matrix(subset, exp_name, filter_models)
        return jsonify(result)

    @app.route("/api/subsets/<subset>/leaderboard/by-source")
    def api_elo_by_source(subset: str):
        """Get ELO rankings grouped by prompt source."""
        exp_name = request.args.get("exp_name", "__all__", type=str)
        result = data_loader.get_elo_by_source(subset, exp_name)
        return jsonify(result)

    @app.route("/api/subsets/<subset>/elo-history")
    def api_elo_history(subset: str):
        """Get ELO history over time."""
        exp_name = request.args.get("exp_name", "__all__", type=str)
        granularity = request.args.get("granularity", "day", type=str)

        # Support model filter (comma-separated)
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        result = data_loader.get_elo_history(subset, exp_name, granularity, filter_models)
        return jsonify(result)

    @app.route("/api/overview/leaderboards")
    def api_overview_leaderboards():
        """Get leaderboard data for all subsets (for Overview page)."""
        result = data_loader.get_all_subsets_leaderboards()
        return jsonify(result)

    @app.route("/api/cross-subset/info")
    def api_cross_subset_info():
        """Get information about models across multiple subsets."""
        subsets_param = request.args.get("subsets", "", type=str)
        if not subsets_param:
            return jsonify({"error": "subsets parameter is required"}), 400

        subsets = [s.strip() for s in subsets_param.split(",") if s.strip()]
        if len(subsets) < 1:
            return jsonify({"error": "At least 1 subset required"}), 400

        result = data_loader.get_cross_subset_info(subsets)
        return jsonify(result)

    @app.route("/api/cross-subset/elo")
    def api_cross_subset_elo():
        """Compute ELO rankings across multiple subsets."""
        subsets_param = request.args.get("subsets", "", type=str)
        if not subsets_param:
            return jsonify({"error": "subsets parameter is required"}), 400

        subsets = [s.strip() for s in subsets_param.split(",") if s.strip()]
        if len(subsets) < 1:
            return jsonify({"error": "At least 1 subset required"}), 400

        exp_name = request.args.get("exp_name", "__all__", type=str)
        model_scope = request.args.get("model_scope", "all", type=str)

        result = data_loader.get_cross_subset_elo(subsets, exp_name, model_scope)
        return jsonify(result)

    # ========== Image Routes ==========

    @app.route("/images/<subset>/<model>/<int:sample_index>")
    def serve_model_image(subset: str, model: str, sample_index: int):
        """Serve model output image."""
        image_path = data_loader.get_image_path(subset, model, sample_index)

        if not image_path or not os.path.isfile(image_path):
            abort(404)

        # Determine mime type
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mimetype = mime_types.get(ext, "image/png")

        return send_file(
            image_path,
            mimetype=mimetype,
            max_age=3600,  # Cache for 1 hour
        )

    @app.route("/images/<subset>/input/<int:sample_index>")
    @app.route("/images/<subset>/input/<int:sample_index>/<int:img_idx>")
    def serve_input_image(subset: str, sample_index: int, img_idx: int = 0):
        """Serve input image from parquet dataset. Supports multiple images via img_idx."""
        image_bytes = data_loader.get_input_image_by_idx(subset, sample_index, img_idx)

        if not image_bytes:
            abort(404)

        return send_file(
            io.BytesIO(image_bytes),
            mimetype="image/png",
            max_age=3600,
        )

    return app


def run_server(
    arena_dir: str,
    data_dir: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    debug: bool = False,
):
    """
    Run the visualization server.

    Args:
        arena_dir: Path to arena directory
        data_dir: Path to data directory
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    print(f"\n{'='*60}")
    print(f"  GenArena Arena Visualizer")
    print(f"{'='*60}")
    print(f"  Arena Dir: {arena_dir}")
    print(f"  Data Dir:  {data_dir}")
    print(f"{'='*60}")
    print(f"  Preloading data (this may take a while)...")
    print(f"{'='*60}\n")

    app = create_app(arena_dir, data_dir)

    print(f"\n{'='*60}")
    print(f"  Server ready: http://{host}:{port}")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=debug, threaded=True)


def create_hf_app(
    arena_dir: str,
    data_dir: str,
    hf_repo: str,
    image_files: list[str],
) -> Flask:
    """
    Create Flask app for HuggingFace Spaces deployment.

    This version uses HF CDN URLs for model output images instead of
    serving them from local filesystem.

    Args:
        arena_dir: Path to arena directory (metadata only, no images)
        data_dir: Path to data directory containing parquet files
        hf_repo: HuggingFace repo ID for image CDN URLs
        image_files: List of image file paths in the HF repo

    Returns:
        Configured Flask app for HF Spaces
    """
    from genarena.visualize.data_loader import HFArenaDataLoader

    # Get the directory containing this file for templates/static
    app_dir = os.path.dirname(os.path.abspath(__file__))

    app = Flask(
        __name__,
        template_folder=os.path.join(app_dir, "templates"),
        static_folder=os.path.join(app_dir, "static"),
    )

    # Store config
    app.config["ARENA_DIR"] = arena_dir
    app.config["DATA_DIR"] = data_dir
    app.config["USE_HF_CDN"] = True
    app.config["HF_REPO"] = hf_repo

    # Create HF data loader
    data_loader = HFArenaDataLoader(arena_dir, data_dir, hf_repo, image_files)

    # ========== Page Routes ==========

    @app.route("/")
    def index():
        """Main page."""
        return render_template("index.html")

    # ========== API Routes ==========
    # Copy all API routes from create_app - they work the same way

    @app.route("/api/subsets")
    def api_subsets():
        """Get list of available subsets."""
        subsets = data_loader.discover_subsets()
        return jsonify({"subsets": subsets})

    @app.route("/api/subsets/<subset>/info")
    def api_subset_info(subset: str):
        """Get information about a subset."""
        info = data_loader.get_subset_info(subset)
        if not info:
            return jsonify({"error": "Subset not found"}), 404

        return jsonify({
            "name": info.name,
            "models": info.models,
            "experiments": info.experiments,
            "total_battles": info.total_battles,
            "min_input_images": info.min_input_images,
            "max_input_images": info.max_input_images,
            "prompt_sources": info.prompt_sources,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/battles")
    def api_battles(subset: str, exp_name: str):
        """Get paginated battle records."""
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 20, type=int)
        result_filter = request.args.get("result", None, type=str)
        consistency = request.args.get("consistent", None, type=str)
        min_images = request.args.get("min_images", None, type=int)
        max_images = request.args.get("max_images", None, type=int)
        prompt_source = request.args.get("prompt_source", None, type=str)

        models_param = request.args.get("models", None, type=str)
        models = None
        if models_param:
            models = [m.strip() for m in models_param.split(",") if m.strip()]

        consistency_filter = None
        if consistency == "true":
            consistency_filter = True
        elif consistency == "false":
            consistency_filter = False

        records, total = data_loader.get_battles(
            subset=subset,
            exp_name=exp_name,
            page=page,
            page_size=page_size,
            models=models,
            result_filter=result_filter,
            consistency_filter=consistency_filter,
            min_images=min_images,
            max_images=max_images,
            prompt_source=prompt_source,
        )

        return jsonify({
            "battles": [r.to_dict() for r in records],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/battles/<path:battle_id>")
    def api_battle_detail(subset: str, exp_name: str, battle_id: str):
        """Get detailed battle record."""
        try:
            parts = battle_id.rsplit(":", 1)
            sample_index = int(parts[1])
            model_part = parts[0]

            if "_vs_" in model_part:
                models = model_part.split("_vs_")
                model_a, model_b = models[0], models[1]
            else:
                return jsonify({"error": "Invalid battle_id format"}), 400
        except (ValueError, IndexError):
            return jsonify({"error": "Invalid battle_id format"}), 400

        record = data_loader.get_battle_detail(
            subset, exp_name, model_a, model_b, sample_index
        )

        if not record:
            return jsonify({"error": "Battle not found"}), 404

        return jsonify(record.to_detail_dict())

    @app.route("/api/subsets/<subset>/stats")
    def api_stats(subset: str):
        """Get statistics for a subset."""
        exp_name = request.args.get("exp_name", None, type=str)
        stats = data_loader.get_stats(subset, exp_name)

        if not stats:
            return jsonify({"error": "Subset not found"}), 404

        return jsonify(stats)

    @app.route("/api/subsets/<subset>/leaderboard")
    def api_elo_leaderboard(subset: str):
        """Get ELO leaderboard for a subset."""
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        leaderboard = data_loader.get_elo_leaderboard(subset, filter_models)
        return jsonify({"leaderboard": leaderboard})

    @app.route("/api/subsets/<subset>/models/<path:model>/stats")
    def api_model_stats(subset: str, model: str):
        """Get detailed statistics for a specific model."""
        exp_name = request.args.get("exp_name", "__all__", type=str)
        stats = data_loader.get_model_vs_stats(subset, model, exp_name)

        if not stats:
            return jsonify({"error": "Model not found"}), 404

        return jsonify(stats)

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/h2h")
    def api_head_to_head(subset: str, exp_name: str):
        """Get head-to-head statistics between two models."""
        model_a = request.args.get("model_a", None, type=str)
        model_b = request.args.get("model_b", None, type=str)

        if not model_a or not model_b:
            return jsonify({"error": "model_a and model_b are required"}), 400

        h2h = data_loader.get_head_to_head(subset, exp_name, model_a, model_b)
        return jsonify(h2h)

    @app.route("/api/subsets/<subset>/samples/<int:sample_index>/input_count")
    def api_input_image_count(subset: str, sample_index: int):
        """Get the number of input images for a sample."""
        count = data_loader.get_input_image_count(subset, sample_index)
        return jsonify({"count": count})

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/samples/<int:sample_index>/all_models")
    def api_sample_all_models(subset: str, exp_name: str, sample_index: int):
        """Get all model outputs for a specific sample."""
        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        stats_scope = request.args.get("stats_scope", "filtered", type=str)

        result = data_loader.get_sample_all_models(
            subset, exp_name, sample_index, filter_models, stats_scope
        )

        if not result:
            return jsonify({"error": "Sample not found"}), 404

        return jsonify(result)

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/samples/<int:sample_index>/models/<path:model>/battles")
    def api_model_battles_for_sample(subset: str, exp_name: str, sample_index: int, model: str):
        """Get all battle records for a specific model on a specific sample."""
        opponents_param = request.args.get("opponents", None, type=str)
        opponent_models = None
        if opponents_param:
            opponent_models = [m.strip() for m in opponents_param.split(",") if m.strip()]

        result = data_loader.get_model_battles_for_sample(
            subset=subset,
            exp_name=exp_name,
            sample_index=sample_index,
            model=model,
            opponent_models=opponent_models,
        )

        return jsonify(result)

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/prompts")
    def api_prompts(subset: str, exp_name: str):
        """Get paginated list of prompts/samples."""
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 10, type=int)
        min_images = request.args.get("min_images", None, type=int)
        max_images = request.args.get("max_images", None, type=int)
        prompt_source = request.args.get("prompt_source", None, type=str)

        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        prompts, total = data_loader.get_prompts(
            subset=subset,
            exp_name=exp_name,
            page=page,
            page_size=page_size,
            min_images=min_images,
            max_images=max_images,
            prompt_source=prompt_source,
            filter_models=filter_models,
        )

        return jsonify({
            "prompts": prompts,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/search")
    def api_search(subset: str, exp_name: str):
        """Search battles by text query."""
        query = request.args.get("q", "", type=str)
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 20, type=int)
        consistency = request.args.get("consistent", None, type=str)

        models_param = request.args.get("models", None, type=str)
        models = None
        if models_param:
            models = [m.strip() for m in models_param.split(",") if m.strip()]

        consistency_filter = None
        if consistency == "true":
            consistency_filter = True
        elif consistency == "false":
            consistency_filter = False

        records, total = data_loader.search_battles(
            subset=subset,
            exp_name=exp_name,
            query=query,
            page=page,
            page_size=page_size,
            models=models,
            consistency_filter=consistency_filter,
        )

        return jsonify({
            "battles": [r.to_dict() for r in records],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "query": query,
        })

    @app.route("/api/subsets/<subset>/experiments/<exp_name>/search/prompts")
    def api_search_prompts(subset: str, exp_name: str):
        """Search prompts by text query."""
        query = request.args.get("q", "", type=str)
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 10, type=int)

        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        prompts, total = data_loader.search_prompts(
            subset=subset,
            exp_name=exp_name,
            query=query,
            page=page,
            page_size=page_size,
            filter_models=filter_models,
        )

        return jsonify({
            "prompts": prompts,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "query": query,
        })

    @app.route("/api/subsets/<subset>/matrix")
    def api_win_rate_matrix(subset: str):
        """Get win rate matrix for all model pairs."""
        exp_name = request.args.get("exp_name", "__all__", type=str)

        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        result = data_loader.get_win_rate_matrix(subset, exp_name, filter_models)
        return jsonify(result)

    @app.route("/api/subsets/<subset>/leaderboard/by-source")
    def api_elo_by_source(subset: str):
        """Get ELO rankings grouped by prompt source."""
        exp_name = request.args.get("exp_name", "__all__", type=str)
        result = data_loader.get_elo_by_source(subset, exp_name)
        return jsonify(result)

    @app.route("/api/subsets/<subset>/elo-history")
    def api_elo_history(subset: str):
        """Get ELO history over time."""
        exp_name = request.args.get("exp_name", "__all__", type=str)
        granularity = request.args.get("granularity", "day", type=str)

        models_param = request.args.get("models", None, type=str)
        filter_models = None
        if models_param:
            filter_models = [m.strip() for m in models_param.split(",") if m.strip()]

        result = data_loader.get_elo_history(subset, exp_name, granularity, filter_models)
        return jsonify(result)

    @app.route("/api/overview/leaderboards")
    def api_overview_leaderboards():
        """Get leaderboard data for all subsets."""
        result = data_loader.get_all_subsets_leaderboards()
        return jsonify(result)

    @app.route("/api/cross-subset/info")
    def api_cross_subset_info():
        """Get information about models across multiple subsets."""
        subsets_param = request.args.get("subsets", "", type=str)
        if not subsets_param:
            return jsonify({"error": "subsets parameter is required"}), 400

        subsets = [s.strip() for s in subsets_param.split(",") if s.strip()]
        if len(subsets) < 1:
            return jsonify({"error": "At least 1 subset required"}), 400

        result = data_loader.get_cross_subset_info(subsets)
        return jsonify(result)

    @app.route("/api/cross-subset/elo")
    def api_cross_subset_elo():
        """Compute ELO rankings across multiple subsets."""
        subsets_param = request.args.get("subsets", "", type=str)
        if not subsets_param:
            return jsonify({"error": "subsets parameter is required"}), 400

        subsets = [s.strip() for s in subsets_param.split(",") if s.strip()]
        if len(subsets) < 1:
            return jsonify({"error": "At least 1 subset required"}), 400

        exp_name = request.args.get("exp_name", "__all__", type=str)
        model_scope = request.args.get("model_scope", "all", type=str)

        result = data_loader.get_cross_subset_elo(subsets, exp_name, model_scope)
        return jsonify(result)

    # ========== Image Routes ==========

    @app.route("/images/<subset>/<model>/<int:sample_index>")
    def serve_model_image(subset: str, model: str, sample_index: int):
        """Redirect to HF CDN for model output images."""
        url = data_loader.get_model_image_url(subset, model, sample_index)
        if url:
            return redirect(url)
        abort(404)

    @app.route("/images/<subset>/input/<int:sample_index>")
    @app.route("/images/<subset>/input/<int:sample_index>/<int:img_idx>")
    def serve_input_image(subset: str, sample_index: int, img_idx: int = 0):
        """Serve input image from parquet dataset."""
        image_bytes = data_loader.get_input_image_by_idx(subset, sample_index, img_idx)

        if not image_bytes:
            abort(404)

        return send_file(
            io.BytesIO(image_bytes),
            mimetype="image/png",
            max_age=3600,
        )

    return app
