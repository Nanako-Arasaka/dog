#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 publishers used by the integration bridge."""

from __future__ import annotations


class RosBridgePublishers:
    def __init__(self, node):
        from std_msgs.msg import String

        self._string_type = String
        self._inspection_pub = node.create_publisher(String, "/inspection/all", 10)
        self._placement_pub = node.create_publisher(String, "/placement/recognized_zone", 10)

    def publish_inspection_all(self, text: str) -> None:
        self._inspection_pub.publish(self._string_type(data=text))

    def publish_placement_zone(self, zone: str) -> None:
        self._placement_pub.publish(self._string_type(data=zone))
