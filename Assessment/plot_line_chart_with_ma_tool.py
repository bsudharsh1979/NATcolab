# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
plot_line_chart_with_ma_tool
============================

NeMo Agent Toolkit (NAT) tool that:

1. Loads sensor data from a JSON file (same as ``plot_line_chart_tool``)
2. Plots the raw data as a line chart (blue)
3. Computes a moving average of the y-axis column
4. Overlays the moving average as a second line (red) on the same chart
5. Saves a JSON file (``moving_average_output.json``) with the raw data and a
   column named ``moving_average``
"""

import json
import logging
import os

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Name of the JSON file written by this tool (graded deliverable)
MOVING_AVERAGE_OUTPUT_FILENAME = "moving_average_output.json"


def verify_json_path(file_path: str, output_folder: str = None) -> str:
    """
    Verify that the input is a valid path to a JSON file.

    Args:
        file_path (str): Path to verify
        output_folder (str): Output folder to use as base for relative paths

    Returns:
        str: Verified (resolved) file path

    Raises:
        ValueError: If input is not a string or not a JSON file
        FileNotFoundError: If file does not exist
    """
    if not isinstance(file_path, str):
        raise ValueError("Input must be a string path to a JSON file")

    if not file_path.lower().endswith('.json'):
        raise ValueError("Input must be a path to a JSON file (ending with .json)")

    # Resolve path relative to output folder if provided
    if output_folder:
        from .plot_utils import resolve_relative_path
        resolved_path = resolve_relative_path(file_path, output_folder)
    else:
        resolved_path = file_path

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"JSON file not found at path: {file_path}")

    try:
        with open(resolved_path, 'r') as f:
            json.load(f)  # Verify file contains valid JSON
    except json.JSONDecodeError:
        raise ValueError(f"File at {file_path} does not contain valid JSON data")

    return resolved_path


class PlotLineChartWithMAToolConfig(FunctionBaseConfig, name="plot_line_chart_with_ma_tool"):
    """
    NeMo Agent Toolkit function to plot a line chart with a moving average overlay.
    """
    output_folder: str = Field(description="The path to the output folder to save plots.",
                               default="./output_data")


@register_function(config_type=PlotLineChartWithMAToolConfig)
async def plot_line_chart_with_ma_tool(
    config: PlotLineChartWithMAToolConfig, builder: Builder
):
    class PlotLineChartWithMAInputSchema(BaseModel):
        data_json_path: str = Field(description="The path to the JSON file containing the data")
        x_axis_column: str = Field(description="The column name for x-axis data", default="time_in_cycles")
        y_axis_column: str = Field(description="The column name for y-axis data", default="RUL")
        plot_title: str = Field(description="The title for the plot", default="Line Chart with Moving Average")
        # ===================== ADD YOUR CODE HERE (1 of 4) =====================
        # Add a window_size parameter so the agent can control the moving-average window.
        window_size: int = Field(description="The window size (number of data points) used to compute the moving average",
                                 default=10)
        # =======================================================================

    from .plot_utils import load_data_from_json, save_plotly_as_png

    async def _response_fn(data_json_path: str,
                           x_axis_column: str,
                           y_axis_column: str,
                           plot_title: str,
                           window_size: int = 10) -> str:
        """
        Process the input message and generate a line chart with a moving average overlay.
        """
        import plotly.graph_objects as go
        import plotly.offline as pyo

        try:
            data_json_path = verify_json_path(data_json_path, config.output_folder)

            # Load data to validate columns exist
            df = load_data_from_json(data_json_path, config.output_folder)
            if df is None or df.empty:
                return "Could not load data or data is empty from the provided JSON file"

            # Check required columns
            required_columns = [x_axis_column, y_axis_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return (f"Data from {data_json_path} must contain columns: {required_columns}. "
                        f"Missing: {missing_columns}")

            # Sanitize the window size (must be a positive integer no larger than the data length)
            try:
                window_size = int(window_size)
            except (TypeError, ValueError):
                window_size = 10
            window_size = max(1, min(window_size, len(df)))

            # Sort by x-axis column for proper line plotting
            df_sorted = df.sort_values(x_axis_column).reset_index(drop=True)

            # ===================== ADD YOUR CODE HERE (2 of 4) =====================
            # Compute the moving average of the y-axis column using pandas.
            df_sorted['moving_average'] = df_sorted[y_axis_column].rolling(window=window_size).mean()
            # =======================================================================

            # Create the figure
            fig = go.Figure()

            # Raw data trace (blue)
            fig.add_trace(go.Scatter(
                x=df_sorted[x_axis_column],
                y=df_sorted[y_axis_column],
                mode='lines+markers',
                name=y_axis_column,
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6, color='#1f77b4'),
                hovertemplate=f'<b>{x_axis_column}</b>: %{{x}}<br>' +
                              f'<b>{y_axis_column}</b>: %{{y:.2f}}<br>' +
                              '<extra></extra>'
            ))

            # ===================== ADD YOUR CODE HERE (3 of 4) =====================
            # Moving average trace (red) overlaid on the same chart.
            fig.add_trace(go.Scatter(
                x=df_sorted[x_axis_column],
                y=df_sorted['moving_average'],
                mode='lines',
                name=f'Moving Average (window={window_size})',
                line=dict(color='red', width=3),
                hovertemplate=f'<b>{x_axis_column}</b>: %{{x}}<br>' +
                              '<b>Moving Average</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))
            # =======================================================================

            # Update layout
            fig.update_layout(
                title=dict(text=plot_title, x=0.5, font=dict(size=16)),
                xaxis=dict(title=dict(text=x_axis_column, font=dict(size=14)),
                           gridcolor='lightgray', gridwidth=0.5),
                yaxis=dict(title=dict(text=y_axis_column, font=dict(size=14)),
                           gridcolor='lightgray', gridwidth=0.5),
                width=650, height=450, plot_bgcolor='white',
                showlegend=True,
                legend=dict(x=1, y=0, xanchor='right', yanchor='bottom',
                            bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1),
                hovermode='closest'
            )

            # Set y-axis range
            y_min = df_sorted[y_axis_column].min()
            y_max = df_sorted[y_axis_column].max()
            y_range = y_max - y_min
            if y_range > 0:
                fig.update_yaxes(range=[y_min - y_range * 0.05, y_max + y_range * 0.05])

            # Save files
            os.makedirs(config.output_folder, exist_ok=True)

            # HTML file
            html_filepath = os.path.join(
                config.output_folder, f"line_chart_with_ma_{x_axis_column}_vs_{y_axis_column}.html")
            html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{plot_title}</title>
    <meta charset="utf-8">
</head>
<body>
    {html_content}
</body>
</html>
"""
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(full_html)
            logger.info(f"Line chart with moving average HTML saved: {html_filepath}")

            # PNG file
            png_filepath = os.path.join(
                config.output_folder, f"line_chart_with_ma_{x_axis_column}_vs_{y_axis_column}.png")
            try:
                png_success = save_plotly_as_png(fig, png_filepath, width=650, height=450)
            except Exception as png_err:  # pragma: no cover - PNG is best-effort
                logger.error(f"PNG generation failed: {png_err}")
                png_success = False
            if not png_success:
                png_filepath = None

            # ===================== ADD YOUR CODE HERE (4 of 4) =====================
            # Save the raw data plus the moving average column to a JSON file.
            output_df = df_sorted[[x_axis_column, y_axis_column, 'moving_average']]
            json_output_path = os.path.join(config.output_folder, MOVING_AVERAGE_OUTPUT_FILENAME)
            output_df.to_json(json_output_path, orient='records', indent=2)
            logger.info(f"Moving average JSON saved: {json_output_path}")
            # =======================================================================

            # Build file information for response (relative paths from output folder)
            html_relpath = os.path.relpath(html_filepath, config.output_folder)
            json_relpath = os.path.relpath(json_output_path, config.output_folder)
            file_info = f"- HTML File: {html_relpath}"
            if png_filepath:
                png_relpath = os.path.relpath(png_filepath, config.output_folder)
                file_info += f"\n- PNG File: {png_relpath}"
            file_info += f"\n- JSON File (raw data + moving_average column): {json_relpath}"

            # Return a clear completion message that the LLM will understand
            return f"""TASK COMPLETED SUCCESSFULLY

Line chart with moving average overlay has been generated and saved in multiple formats.

Chart Details:
- Type: Line chart with markers (Plotly) + moving average overlay (red line)
- X-axis: {x_axis_column}
- Y-axis: {y_axis_column}
- Moving average window size: {window_size}
- Title: {plot_title}
{file_info}

✅ CHART GENERATION COMPLETE - NO FURTHER ACTION NEEDED"""

        except FileNotFoundError as e:
            error_msg = f"Required data file ('{data_json_path}') not found for line chart: {e}"
            logger.error(error_msg)
            return error_msg
        except KeyError as ke:
            error_msg = f"Missing required columns in '{data_json_path}' for line chart: {ke}"
            logger.error(error_msg)
            return error_msg
        except ValueError as ve:
            error_msg = f"Data validation error for line chart: {ve}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error generating line chart with moving average: {e}"
            logger.error(error_msg)
            return error_msg

    prompt = """
    Generate an interactive line chart from JSON data using Plotly, with a moving average overlay.

    Input:
    - data_json_path: Path to the JSON file containing the data
    - x_axis_column: Column name for x-axis data
    - y_axis_column: Column name for y-axis data
    - plot_title: Title for the plot
    - window_size: Window size (number of points) for the moving average (default 10)

    Output:
    - HTML file containing the interactive line chart (raw data in blue, moving average in red)
    - PNG file containing the static line chart
    - JSON file (moving_average_output.json) containing the raw data plus a 'moving_average' column
    """
    yield FunctionInfo.from_fn(_response_fn,
                               input_schema=PlotLineChartWithMAInputSchema,
                               description=prompt)
    try:
        pass
    except GeneratorExit:
        logger.info("Plot line chart with moving average function exited early!")
    finally:
        logger.info("Cleaning up plot_line_chart_with_ma_tool workflow.")
