from catch_analysis_tools.app.astrometry_readiness.get_astrometry_readiness_status import (  # noqa: E501
    get_astrometry_readiness_status,
)
from catch_analysis_tools.app.astrometry_readiness.prepare_astrometry_data import (
    prepare_astrometry_data,
)


def main():
    prepare_astrometry_data()
    print(get_astrometry_readiness_status())


if __name__ == "__main__":
    main()
