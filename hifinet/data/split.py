def split(data, temp):
    match temp:
        case 0:
            return (
                data[data["datetime"] < "2023-08-01"],
                data[
                    (data["datetime"] >= "2023-08-01")
                    | (data["datetime"] < "2023-10-01")
                ],
                data[
                    (data["datetime"] >= "2023-10-01")
                    | (data["datetime"] < "2023-11-01")
                ],
            )
        case 1:
            return (
                data[
                    (data["datetime"] >= "2023-02-01")
                    | (data["datetime"] < "2023-09-01")
                ],
                data[
                    (data["datetime"] >= "2023-09-01")
                    | (data["datetime"] < "2023-11-01")
                ],
                data[
                    (data["datetime"] >= "2023-11-01")
                    | (data["datetime"] < "2023-12-01")
                ],
            )
        case 2:
            return (
                data[
                    (data["datetime"] >= "2023-03-01")
                    | (data["datetime"] < "2023-10-01")
                ],
                data[
                    (data["datetime"] >= "2023-10-01")
                    | (data["datetime"] < "2023-12-01")
                ],
                data[data["datetime"] >= "2023-12-03"],
            )
        case _:
            raise NotImplementedError
