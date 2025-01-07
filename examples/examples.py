all_examples = [
            {
                "input_image": None,
                "input_mask": None,
                "input_reference_image": "assets/samples/portrait/human_1.jpg",
                "save_path": "examples/outputs/portrait_human_1.jpg",
                "instruction": "Maintain the facial features, A girl is wearing a neat police uniform and sporting a badge. She is smiling with a friendly and confident demeanor. The background is blurred, featuring a cartoon logo.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 4194866942,
                "repainting_scale": 1.0,
                "task_type": "portrait",
                "edit_type": "repainting"
            },
            {
                "input_image": None,
                "input_mask": None,
                "input_reference_image": "assets/samples/subject/subject_1.jpg",
                "save_path": "examples/outputs/subject_subject_1.jpg",
                "instruction": "Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug, alongside a steaming cup of coffee on a cozy cafe table.",
                "output_h": 1024,
                "output_w": 1024,
                "seed": 2935362780,
                "repainting_scale": 1.0,
                "task_type": "subject",
                "edit_type": "repainting"
            },
            {
                "input_image": "assets/samples/local/local_1.webp",
                "input_mask":  "assets/samples/local/local_1_m.webp",
                "input_reference_image": None,
                "save_path": "examples/outputs/local_local_1.jpg",
                "instruction": "By referencing the mask, restore a partial image from the doodle {image} that aligns with the textual explanation: \"1 white old owl\".",
                "output_h": -1,
                "output_w": -1,
                "seed": 1159797084,
                "repainting_scale": 0.5,
                "task_type": "local_editing",
                "edit_type": "contour_repainting"
            },
            {
                "input_image": "assets/samples/application/photo_editing/1_1_edit.png",
                "input_mask": "assets/samples/application/photo_editing/1_1_m.png",
                "input_reference_image": "assets/samples/application/photo_editing/1_ref.png",
                "save_path": "examples/outputs/photo_editing_1.jpg",
                "instruction": "The item is put on the ground.",
                "output_h": -1,
                "output_w": -1,
                "seed": 2072028954,
                "repainting_scale": 1.0,
                "task_type": "subject",
                "edit_type": "repainting"
            },
            {
                "input_image": "assets/samples/application/logo_paste/1_1_edit.png",
                "input_mask": "assets/samples/application/logo_paste/1_1_m.png",
                "input_reference_image": "assets/samples/application/logo_paste/1_ref.png",
                "save_path": "examples/outputs/logo_paste_1.jpg",
                "instruction": "The logo is printed on the headphones.",
                "output_h": -1,
                "output_w": -1,
                "seed": 934582264,
                "repainting_scale": 1.0,
                "task_type": "subject",
                "edit_type": "repainting"
            },
            {
                "input_image": "assets/samples/application/movie_poster/1_1_edit.png",
                "input_mask": "assets/samples/application/movie_poster/1_1_m.png",
                "input_reference_image": "assets/samples/application/movie_poster/1_ref.png",
                "save_path": "examples/outputs/movie_poster_1.jpg",
                "instruction": "The man is facing the camera and is smiling.",
                "output_h": -1,
                "output_w": -1,
                "seed": 988183236,
                "repainting_scale": 1.0,
                "task_type": "portrait",
                "edit_type": "repainting"
            }

        ]