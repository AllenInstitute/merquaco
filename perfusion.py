import numpy as np
from typing import Union
from pathlib import Path


@staticmethod
def analyze_flow(log_file_path: Union[str, Path]):
    """
    Extract and analyze flow rate information from log file

    Parameters
    ----------
    log_file_path : str or Path
        Path to log file

    Returns
    -------
    tuple
        hrs_after_start : np.ndarray
            Time array for experiment duration
        flow_rates : np.ndarray
            Flow rates at corresponding times in hrs_after_start
        median_flow_times : np.ndarray
            Array of flow time for each cartridge position and time stamp
        median_flow_rates : np.ndarray
            Array of median flow rates for each cartridge position and time stamp
    """
    flow_times = []
    flow_list = []
    data_list = []

    with open(log_file_path, 'r') as f:
        log = f.read()

        print('Beginning fluidics log file analysis')

        # Find locations of timestamps, cartridge positions, flow rates
        cartridge_locations = [i for i in range(len(log)) if log.startswith('cartridge_position=', i)]
        cartridge_locations = np.asarray(cartridge_locations, dtype=int) + 19
        time_locations = cartridge_locations - 67
        time_locations = np.asarray(time_locations, dtype=int)

        if len(cartridge_locations) == 0:
            cartridge_locations = [i for i in range(len(log)) if log.startswith('cartridge_cmd_position=', i)]
            cartridge_locations = np.asarray(cartridge_locations, dtype=int) + 23
            time_locations = cartridge_locations - 71
            time_locations = np.asarray(time_locations, dtype=int)

        if len(cartridge_locations) > 0:
            flow_locations = []
            for ii in range(len(cartridge_locations)):
                line_end = log[cartridge_locations[ii]:].find('\n')
                flow_locations.append(cartridge_locations[ii] +
                                        log[cartridge_locations[ii]:cartridge_locations[ii] +
                                            line_end].find('flow='))
            flow_locations = np.asarray(flow_locations, dtype=int)

            # Create list of times, cartridge positions, flow rates
            times = [('0', '0', '0', '0', '0', '0')]
            sec_after_start = []
            flow_rates = []
            cartridge_positions = []
            for ii in range(len(cartridge_locations)):
                year = int(log[time_locations[ii]:time_locations[ii]+4])
                month = int(log[time_locations[ii]+5:time_locations[ii]+7])
                day = int(log[time_locations[ii]+8:time_locations[ii]+10])
                hrs = int(log[time_locations[ii]+11:time_locations[ii]+13])
                mins = int(log[time_locations[ii]+14:time_locations[ii]+16])
                sec = log[time_locations[ii]+17:time_locations[ii]+22]
                sec = float(sec.replace(',', '.'))
                times.append((year, month, day, hrs, mins, sec))

                # Calculate seconds after first time stamp
                if year != times[-1][0]:
                    month = month + 12
                if month != times[-1][1]:
                    day = day + times[-1][1]
                time_after = (((day - times[1][2]) * 24 + (hrs - times[1][3])) *
                                60 + (mins - times[1][4])) * 60 + sec - times[1][5]
                sec_after_start.append(time_after)

                # Flow rate
                flow_end = log[flow_locations[ii]:].find(',')
                flow = log[flow_locations[ii]+5:flow_locations[ii]+flow_end]
                flow_rates.append(flow)

                # Cartridge position
                cartridge = log[cartridge_locations[ii]:cartridge_locations[ii]+6]
                cartridge_positions.append(cartridge)

            hrs_after_start = np.asarray(sec_after_start) / 3600
            flow_rates = np.asarray(flow_rates, dtype=(float))
            cartridge_positions = np.asarray(cartridge_positions, dtype=(float))

            # Calculate mean flow for each new cartridge position
            cartridge_changes = np.where(cartridge_positions[:-1] != cartridge_positions[1:])[0]
            cartridge_changes = np.insert(cartridge_changes, 0, 0)
            median_flow_rates = []
            median_flow_times = []

            for ii in range(len(cartridge_changes[:-1])):
                median_flow_rates.append(np.median(flow_rates[cartridge_changes[ii]:cartridge_changes[ii+1]]))
                median_flow_times.append(np.median(hrs_after_start[cartridge_changes[ii]:cartridge_changes[ii+1]]))
            median_flow_rates = np.asarray(median_flow_rates, dtype=float)
            median_flow_times = np.asarray(median_flow_times, dtype=float)

            # Low flow warning, if mean flow for any position is too low
            if np.min(median_flow_rates[1:-2]) < 0.8:
                print('Low flow warning')
                print(np.min(median_flow_rates), median_flow_rates)
                print(' ')

            data_list.append(log[:19])
            flow_times.append(median_flow_times)
            flow_list.append(median_flow_rates)

            return hrs_after_start, flow_rates, median_flow_times, median_flow_rates

        else:
            print('No flow measurements')
            print(' ')
