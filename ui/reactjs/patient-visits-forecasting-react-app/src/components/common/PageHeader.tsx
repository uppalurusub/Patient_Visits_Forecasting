import type { ReactNode } from "react";

import {
    Box,
    Breadcrumbs,
    Link,
    Stack,
    Typography
} from "@mui/material";

import NavigateNextIcon from "@mui/icons-material/NavigateNext";

export interface BreadcrumbItem {

    label: string;

    href?: string;

}

interface PageHeaderProps {

    title: string;

    subtitle?: string;

    icon?: ReactNode;

    actions?: ReactNode;

    breadcrumbs?: BreadcrumbItem[];

}

const PageHeader = ({

    title,

    subtitle,

    icon,

    actions,

    breadcrumbs

}: PageHeaderProps) => {

    return (

        <Box mb={4}>

            {

                breadcrumbs && breadcrumbs.length > 0 && (

                    <Breadcrumbs
                        separator={<NavigateNextIcon fontSize="small" />}
                        sx={{ mb: 2 }}
                    >

                        {

                            breadcrumbs.map((item, index) => (

                                item.href ? (

                                    <Link
                                        key={index}
                                        underline="hover"
                                        color="inherit"
                                        href={item.href}
                                    >

                                        {item.label}

                                    </Link>

                                ) : (

                                    <Typography
                                        key={index}
                                        color="text.primary"
                                    >

                                        {item.label}

                                    </Typography>

                                )

                            ))

                        }

                    </Breadcrumbs>

                )

            }

            <Stack
                direction={{ xs: "column", md: "row" }}
                justifyContent="space-between"
                alignItems={{ xs: "flex-start", md: "center" }}
                spacing={2}
            >

                <Stack
                    direction="row"
                    spacing={2}
                    alignItems="center"
                >

                    {icon}

                    <Box>

                        <Typography

                            variant="h4"

                            fontWeight={700}

                        >

                            {title}

                        </Typography>

                        {

                            subtitle && (

                                <Typography

                                    variant="body1"

                                    color="text.secondary"

                                >

                                    {subtitle}

                                </Typography>

                            )

                        }

                    </Box>

                </Stack>

                {actions}

            </Stack>

        </Box>

    );

};

export default PageHeader;