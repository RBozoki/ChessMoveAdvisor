Deep Learning Project: Chess Move Prediction

Author: Raphaël Bozoki
Advisor: [Advisor's Name]
Description

This project aims to provide an application that identifies the best chess move in a given position. Users can upload an image of a chessboard, and the application returns the optimal move based on automated analysis.
Roadmap

    Literature Review: Study existing research to guide development.
    Develop a Computer Vision Model: Detect and analyze the position of chess pieces.
    Build an Application: Integrate the model into a user-friendly interface.

Dataset

    Sources: Positions played by Magnus Carlsen and randomly generated positions.
    Images:
        1576 for training
        146 for validation
        342 for testing
    Augmentations: Positions with various angles and lighting conditions.
    Format: FEN notation to represent board states.

Developed Models

    Board Localization: Detects the edges of the chessboard.
        Algorithms:
            Edge Detection (e.g., Canny)
            Line Intersections (e.g., Hough Lines)
            Perspective Transformation
    Binary Classification: Identifies if a square is occupied.
        Architecture: CNN with ReLU, Max Pooling, and Softmax layers.
    Multi-class Classification: Identifies the type of piece on an occupied square.

Position Evaluation

    Stockfish Integration: Evaluates board positions and suggests the best move.
    Handles edge cases, such as positions never encountered in play.

The Application

A simple and user-friendly interface allowing users to upload images and receive move suggestions.
